import gradio as gr
import os
import logging
import time
import threading
import re  
from typing import List, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import modules
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from summarizer import DocumentSummarizer
from serve import FileServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGDocumentApp:
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.summarizer = DocumentSummarizer()
        self.file_server = FileServer()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Start file server
        self.file_server.start_server()
        
        # App state
        self.current_document = None
        self.is_processing = False
        self.is_ready_for_chat = False
        self.processing_progress = "Ready"

    def validate_api_key(self) -> Tuple[str, bool]:
        """Validate Gemini API key"""
        try:
            if not self.config.GEMINI_API_KEY:
                return "❌ API key not found", False
            
            is_valid = self.llm_handler.validate_api_key()
            return ("✅ API key is valid!" if is_valid else "❌ API key is invalid", is_valid)
        except Exception as e:
            return f"❌ API key error: {str(e)}", False

    def process_document_async(self):
        """Process document in background"""
        def background_task():
            try:
                if not self.current_document:
                    return
                
                self.is_processing = True
                self.processing_progress = "🔄 Processing..."
                
                chunks = self.doc_processor.process_document(self.current_document)
                self.processing_progress = f"🧠 Creating embeddings for {len(chunks)} chunks..."
                
                self.vector_store.add_documents(chunks)
                
                self.is_ready_for_chat = True
                self.is_processing = False
                self.processing_progress = "✅ Ready for intelligent chat!"
                
                logger.info("Background processing completed")
                
            except Exception as e:
                self.is_processing = False
                self.is_ready_for_chat = False
                self.processing_progress = f"❌ Processing failed: {str(e)}"
        
        thread = threading.Thread(target=background_task)
        thread.daemon = True
        thread.start()

    def upload_document(self, file) -> Tuple[str, str, str]:
        """Upload document with smart processing"""
        try:
            if file is None:
                return "No file selected", "", ""
            
            # Check file format
            file_ext = Path(file.name).suffix.lower()
            if file_ext not in self.config.SUPPORTED_FORMATS:
                return f"Unsupported format: {file_ext}", "", ""
            
            # Save file
            filename = os.path.basename(file.name)
            destination = self.doc_processor.save_uploaded_file(file, filename)
            doc_info = self.doc_processor.get_document_info(destination)
            
            # Update state
            self.current_document = destination
            self.is_ready_for_chat = False
            self.is_processing = False
            
            # Clear chat history
            if hasattr(self.llm_handler, 'clear_conversation_history'):
                self.llm_handler.clear_conversation_history()
            
            # Smart processing based on file size
            file_size_mb = doc_info['file_size'] / (1024 * 1024)
            
            if file_size_mb <= 2:  # Small files - instant processing
                try:
                    chunks = self.doc_processor.process_document(self.current_document)
                    self.vector_store.add_documents(chunks)
                    
                    self.is_ready_for_chat = True
                    status = "✅ Processed instantly!"
                except Exception as e:
                    logger.warning(f"Instant processing failed, falling back to background: {e}")
                    status = "🔄 Processing in background..."
                    self.process_document_async()
            else:  # Large files - background processing
                status = "🔄 Processing in background..."
                self.process_document_async()
            
            return f"✅ Document uploaded successfully! {status}", filename, self.file_server.get_file_url(filename)
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return f"❌ Upload error: {str(e)}", "", ""

    def generate_summary(self, length_choice="Standard", progress=gr.Progress()) -> str:
        try:
            if not self.current_document:
                return "❌ No document uploaded. Please upload a document first."

            filename = os.path.basename(self.current_document)
            
            # Enhanced progress steps with better descriptions
            progress(0.1, desc="🔍 Analyzing document structure...")
            time.sleep(0.5)
            
            progress(0.3, desc="📖 Reading document content...")
            time.sleep(0.5)
            
            mode_map = {"Short": "short", "Standard": "standard", "Detailed": "detailed"}
            mode = mode_map.get(length_choice, "standard")
            
            progress(0.5, desc="🤖 Generating intelligent summary...")
            time.sleep(0.5)
            
            summary_data = self.summarizer.generate_summary(
                self.current_document,
                length_mode=mode
            )
            
            progress(0.8, desc="✨ Finalizing summary...")
            time.sleep(0.3)
            
            if not summary_data or 'summary' not in summary_data:
                progress(1.0, desc="❌ Failed")
                return f"❌ Summary generation failed. Debug: {summary_data}"

            # FIX: Ensure complete summaries for short mode
            summary_text = summary_data['summary']
            
            # If it's short mode and seems incomplete, get a better summary
            if mode == "short":
                # Get standard summary first, then make it short and complete
                full_summary_data = self.summarizer.generate_summary(
                    self.current_document,
                    length_mode="standard"
                )
                
                if full_summary_data and 'summary' in full_summary_data:
                    full_text = full_summary_data['summary']
                    
                    # Take first few sentences but ensure completion
                    sentences = full_text.split('. ')
                    
                    # For short mode, take 3-4 complete sentences
                    if len(sentences) >= 4:
                        summary_text = '. '.join(sentences[:4])
                        if not summary_text.endswith('.'):
                            summary_text += '.'
                    else:
                        # If not enough sentences, use the full summary
                        summary_text = full_text
                        
                    # Ensure proper ending
                    if not summary_text.rstrip().endswith(('.', '!', '?')):
                        summary_text = summary_text.rstrip() + '.'
                else:
                    # Fallback to original if full summary fails
                    if not summary_text.rstrip().endswith(('.', '!', '?')):
                        summary_text = summary_text.rstrip() + '.'

            progress(1.0, desc="✅ Summary complete!")
            return summary_text
            
        except Exception as e:
            progress(1.0, desc="❌ Error occurred")
            return f"❌ Error: {str(e)}"

    def analyze_question_type(self, question_lower: str) -> str:
        """IMPROVED: Intelligently categorize any question type for better responses"""
        
        # Dynamic bullet point detection - extract number if specified
        bullet_patterns = ['list', 'give me', 'provide', 'bullet points', 'key points', 'main points', 'important points']
        if any(pattern in question_lower for pattern in bullet_patterns):
            # Extract number from question if present
            numbers = re.findall(r'\b(\d{1,2})\b', question_lower)  # Find numbers 1-99
            if numbers:
                return f"bullet_points_{numbers[0]}"  # e.g., "bullet_points_7"
            else:
                return "bullet_points_5"  # Default to 5 if no number specified
        
        # Character questions - SPECIFIC patterns first
        elif any(word in question_lower for word in ['main character', 'protagonist', 'who is the main', 'central character']):
            return "character_analysis"
        
        # Author questions  
        elif any(word in question_lower for word in ['who is the author', 'author of', 'written by', 'who wrote']):
            return "direct_answer"
            
        # Fee/cost questions
        elif any(word in question_lower for word in ['fee', 'cost', 'price', 'how much', 'charges']):
            return "direct_answer"
            
        # Target audience questions
        elif any(word in question_lower for word in ['who is this for', 'target audience', 'who can apply', 'eligibility']):
            return "direct_answer"
        
        # Direct answer questions (who, what, when, where, which)
        elif any(word in question_lower for word in ['who is', 'who are', 'who was', 'what is', 'what are', 'what was', 
                                                    'when is', 'when did', 'when was', 'where is', 'where did', 
                                                    'which is', 'which are', 'how many', 'how much']):
            return "direct_answer"
        
        # Analytical questions
        elif any(word in question_lower for word in ['moral', 'lesson', 'theme', 'message', 'significance', 
                                                    'purpose', 'meaning', 'symbolize', 'represent', 'important',
                                                    'why is', 'why does', 'why did']):
            return "analytical"
        
        else:
            return "comprehensive"
        
    def detect_response_length(self, message: str) -> str:
        """Enhanced: Detect desired response length from user message"""
        message_lower = message.lower()
        
        # Ultra short responses (1-2 sentences)
        ultra_short_keywords = ['very short', 'ultra short', 'one sentence', 'just briefly', 'in short']
        if any(keyword in message_lower for keyword in ultra_short_keywords):
            return "ultra_short"
        
        # Short responses (2-3 sentences / 1 small paragraph)
        short_keywords = ['short summary', 'concise', 'quickly', 'short answer']
        if any(keyword in message_lower for keyword in short_keywords):
            return "short"
        
        # Brief responses (1 decent paragraph)
        brief_keywords = ['briefly', 'brief', 'quick overview', 'brief explanation']
        if any(keyword in message_lower for keyword in brief_keywords):
            return "brief"
        
        # Medium responses (2-3 paragraphs)
        medium_keywords = ['medium', 'moderate', 'standard', 'normal length']
        if any(keyword in message_lower for keyword in medium_keywords):
            return "medium"
        
        # Long responses (3-4 paragraphs)
        long_keywords = ['long', 'detailed', 'comprehensive', 'in depth', 'elaborate']
        if any(keyword in message_lower for keyword in long_keywords):
            return "long"
        
        # Very detailed responses (5+ paragraphs)
        very_long_keywords = ['very detailed', 'very long', 'exhaustive', 'complete analysis', 'full explanation']
        if any(keyword in message_lower for keyword in very_long_keywords):
            return "very_long"
        
        # Default to medium if no length specified
        return "medium"

    def detect_formatting_preferences(self, message: str) -> dict:
        """Detect requested format from user message"""
        msg = message.lower()
        return {
            'bullet_points': any(k in msg for k in ['points', 'bullet', 'list important', 'key points', 'in points']),
            'numbered_list': any(k in msg for k in ['numbered', 'list 5', 'give 5', '5 important', 'top 5', 'steps']),
            'paragraph_format': any(k in msg for k in ['summary', 'overview', 'explain', 'tell me about'])
        }

    def chat_with_document(self, message: str, history: List) -> Tuple[str, List]:
        """Enhanced chat with dynamic length control and intelligent question handling"""
        try:
            if not self.current_document:
                history.append({"role": "assistant", "content": "❌ No document uploaded"})
                return "", history

            current_filename = os.path.basename(self.current_document)
            message_lower = message.lower()
            
            # Detect desired response length
            desired_length = self.detect_response_length(message)
            
            # Length-specific instructions for the LLM
            length_instructions = {
                "ultra_short": "Respond in 1-2 sentences only. Be extremely concise.",
                "short": "Respond in 2-3 sentences or one small paragraph. Be concise but clear.",
                "brief": "Respond in one decent paragraph (4-6 sentences). Provide a good overview.",
                "medium": "Respond in 2-3 paragraphs. Provide balanced detail and explanation.",
                "long": "Respond in 3-4 paragraphs. Provide detailed analysis and explanation.",
                "very_long": "Respond in 4+ paragraphs. Provide comprehensive, detailed analysis."
            }

            # More specific summary detection
            is_summary_request = (
                message_lower.startswith('summarize') or 
                'summarize the document' in message_lower or
                'summary of the document' in message_lower or
                'what is this about' in message_lower or
                'briefly summarize' in message_lower or
                'brief summary' in message_lower
            )
            wants_bullets = any(k in message_lower for k in ['points', 'list 5', 'give 5', 'key points', '5 important', 'bullet'])

            # Handle summary requests with length control
            if is_summary_request:
                try:
                    # Map length to summarizer modes
                    summarizer_mode = {
                        "ultra_short": "short",
                        "short": "short", 
                        "brief": "standard",
                        "medium": "standard",
                        "long": "detailed",
                        "very_long": "detailed"
                    }
                    
                    mode = summarizer_mode.get(desired_length, "standard")
                    summary_data = self.summarizer.generate_summary(self.current_document, length_mode=mode)
                    
                    # Post-process based on desired length with completion check
                    if desired_length == "ultra_short":
                        # Take just the first sentence or two but ensure completion
                        sentences = summary_data['summary'].split('. ')
                        response = '. '.join(sentences[:2])
                        if not response.endswith('.'):
                            response += '.'
                    elif desired_length == "brief":
                        # Ensure it's one good paragraph and complete
                        response = summary_data['summary']
                        if not response.rstrip().endswith(('.', '!', '?')):
                            response = response.rstrip() + '.'
                    elif desired_length == "short":
                        # For short responses, ensure completeness
                        sentences = summary_data['summary'].split('. ')
                        if len(sentences) >= 3:
                            response = '. '.join(sentences[:3])
                            if not response.endswith('.'):
                                response += '.'
                        else:
                            response = summary_data['summary']
                            if not response.rstrip().endswith(('.', '!', '?')):
                                response = response.rstrip() + '.'
                    elif desired_length in ["long", "very_long"]:
                        # For longer responses, use full content
                        response = summary_data['summary']
                    else:
                        response = summary_data['summary']
                    
                    if wants_bullets:
                        # Convert to bullet points
                        sentences = response.split('. ')
                        points = []
                        for sentence in sentences[:3 if desired_length == "ultra_short" else 6]:
                            sentence = sentence.strip()
                            if len(sentence) > 15:
                                if not sentence.endswith('.'):
                                    sentence += '.'
                                points.append(f"• {sentence}")
                        response = '\n\n'.join(points)

                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response})
                    return "", history
                    
                except Exception as e:
                    logger.error(f"Summary error: {str(e)}")

            # INTELLIGENT RAG search for any question type with length control
            if self.is_ready_for_chat:
                try:
                    chunks = self.vector_store.search_similar_chunks(message, n_results=20)
                    filtered = [c for c in chunks if c['metadata'].get('filename') == current_filename]

                    if filtered:
                        context = filtered[:10]
                        
                        # INTELLIGENT QUESTION ANALYSIS
                        question_type = self.analyze_question_type(message_lower)
                        print(f"DEBUG: Question type detected: '{question_type}' for question: '{message}'")
                        
                        # Get length instruction
                        length_instruction = length_instructions[desired_length]
                        
                        # SMARTER PROMPTS FOR EACH QUESTION TYPE with length control
                        if question_type.startswith("bullet_points_"):
                            # Extract the number of bullet points requested
                            num_points = question_type.split("_")[-1]
                            enhanced_prompt = f"""QUESTION: {message}

INSTRUCTION: List exactly {num_points} key points in clean bullet format. Each point should be one clear sentence. {length_instruction}

DOCUMENT CONTEXT: {chr(10).join([chunk['content'][:200] for chunk in context[:8]])}

FORMAT:
• First key point
• Second key point
• Third key point
(Continue for {num_points} total points)

Provide exactly {num_points} bullet points:"""

                        elif question_type == "character_analysis":
                            enhanced_prompt = f"""QUESTION: {message}

INSTRUCTION: Identify WHO the main character is. {length_instruction} Do NOT retell the plot.

DOCUMENT CONTEXT: {chr(10).join([chunk['content'][:200] for chunk in context[:3]])}

ANSWER FORMAT: "The main character is [name/description] - [brief significance]":"""

                        elif question_type == "direct_answer":
                            # Enhanced context search for author questions
                            if any(word in message_lower for word in ['author', 'writer', 'written by', 'who wrote']):
                                # Search more chunks for author information
                                context_extended = filtered[:20]  # More context for author searches
                                enhanced_prompt = f"""QUESTION: {message}

INSTRUCTION: Search the document carefully for author information. If you find the author's name anywhere in the document, state it directly like "The author is [Name]" or "[Name] is the author of this story/document". {length_instruction}

Look for author names in:
- Title pages
- Headers or footers  
- Author attribution sections
- Suggested reading sections that mention the current work's author
- Any biographical information

DOCUMENT CONTEXT: {chr(10).join([chunk['content'][:300] for chunk in context_extended[:8]])}

DIRECT ANSWER (If author found, state clearly; if not found, say author not mentioned in available text):"""
                            else:
                                enhanced_prompt = f"""QUESTION: {message}

INSTRUCTION: Give a direct, specific answer. {length_instruction} Do NOT provide a full summary.

DOCUMENT CONTEXT: {chr(10).join([chunk['content'][:200] for chunk in context[:3]])}

DIRECT ANSWER:"""

                        elif question_type == "analytical":
                            enhanced_prompt = f"""QUESTION: {message}

INSTRUCTION: Analyze the deeper meaning/theme. {length_instruction} Focus on interpretation, not plot summary.

DOCUMENT CONTEXT: {chr(10).join([chunk['content'][:300] for chunk in context[:4]])}

ANALYTICAL ANSWER:"""

                        else:
                            enhanced_prompt = f"""QUESTION: {message}

INSTRUCTION: Answer this question directly based on the document. {length_instruction} Be specific and concise.

DOCUMENT CONTEXT: {chr(10).join([chunk['content'][:250] for chunk in context[:5]])}

FOCUSED ANSWER:"""

                        # Pass the enhanced prompt directly to the LLM handler
                        raw_response = self.llm_handler.generate_response_with_memory(enhanced_prompt, context, current_filename)
                        
                        # IMPROVED BULLET POINT PROCESSING - FIX FOR TEXT TRUNCATION
                        if question_type.startswith("bullet_points_"):
                            num_points = int(question_type.split("_")[-1])
                            lines = raw_response.split('\n')
                            clean_bullets = []
                            
                            # First pass: Extract existing bullets
                            for line in lines:
                                line = line.strip()
                                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                                    # FIX: Don't strip the first character if it's part of the content
                                    clean_line = line[1:].strip() if line.startswith(('•', '-', '*')) else line.strip()
                                    if clean_line and len(clean_line) > 10:
                                        clean_bullets.append(f"• {clean_line}")
                                        if len(clean_bullets) >= num_points:
                                            break
                            
                            # Second pass: If not enough bullets, split all meaningful sentences
                            if len(clean_bullets) < num_points:
                                # Clean text and split into sentences
                                clean_text = raw_response.replace('\n', ' ')
                                # Remove bullet markers without destroying content
                                clean_text = re.sub(r'^[•\-\*]\s*', '', clean_text, flags=re.MULTILINE)
                                sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 15]
                                
                                for sentence in sentences:
                                    if len(clean_bullets) < num_points:
                                        # Avoid duplicates by checking if sentence already exists
                                        sentence_clean = sentence.strip()
                                        if not any(sentence_clean in bullet for bullet in clean_bullets):
                                            clean_bullets.append(f"• {sentence_clean}.")
                            
                            response = '\n\n'.join(clean_bullets[:num_points])  # Ensure exact count with proper spacing
                        
                        # Handle general bullet point requests (like "explain in points")
                        elif wants_bullets or any(word in message_lower for word in ['in points', 'explain in points', 'key points']):
                            # Split response into sentences and convert to bullets
                            sentences = [s.strip() for s in raw_response.replace('\n', '. ').split('.') if len(s.strip()) > 15]
                            clean_bullets = []
                            
                            for sentence in sentences[:8]:  # Limit to reasonable number
                                sentence = sentence.strip()
                                if sentence:
                                    # FIX: Better cleaning without truncating content
                                    clean_sentence = re.sub(r'^[•\-\*]\s*', '', sentence).strip()
                                    if clean_sentence and len(clean_sentence) > 10:
                                        if not clean_sentence.endswith('.'):
                                            clean_sentence += '.'
                                        clean_bullets.append(f"• {clean_sentence}")
                            
                            response = '\n\n'.join(clean_bullets)
                            
                        else:
                            response = self.llm_handler.format_bullet_response(raw_response)
                        
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": response})
                        return "", history
                        
                except Exception as e:
                    logger.error(f"RAG error: {str(e)}")

            # Fallback using summary with length control
            try:
                mode = "short" if desired_length in ["ultra_short", "short"] else "standard"
                summary_data = self.summarizer.generate_summary(self.current_document, length_mode=mode)
                
                # Apply length control to fallback
                if desired_length == "ultra_short":
                    sentences = summary_data['summary'].split('. ')
                    response = '. '.join(sentences[:2]) + '.'
                else:
                    response = summary_data['summary']
                
                if wants_bullets:
                    sentences = response.split('. ')
                    points = []
                    for sentence in sentences[:3 if desired_length == "ultra_short" else 6]:
                        sentence = sentence.strip()
                        if len(sentence) > 15:
                            if not sentence.endswith('.'):
                                sentence += '.'
                            points.append(f"• {sentence}")
                    response = '\n\n'.join(points)
                
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history
                
            except Exception as e:
                history.append({"role": "assistant", "content": f"❌ Unable to respond: {str(e)}"})
                return "", history

        except Exception as e:
            history.append({"role": "assistant", "content": f"❌ Chat error: {str(e)}"})
            return "", history

    def clear_chat_history(self) -> List:
        """Clear chat history"""
        self.llm_handler.clear_conversation_history()
        return []

    def get_processing_status(self) -> str:
        """Get processing status"""
        if self.is_ready_for_chat:
            return "🚀 Ready for intelligent chat!"
        elif self.is_processing:
            return f"⏳ {self.processing_progress}"
        else:
            return "💭 Upload a document to begin"

    def get_collection_info(self) -> str:
        """Get database info with better formatting"""
        try:
            info = self.vector_store.get_collection_info()
            if 'error' in info:
                return f"❌ Error: {info['error']}"
            
            total_chunks = info.get('total_chunks', 0)
            total_docs = info.get('total_documents', 0)
            documents = info.get('documents', [])
            
            # Create header
            header = f"""📊 **Vector Database Status**

**Total Chunks:** {total_chunks:,}  
**Total Documents:** {total_docs}

---

### 📄 Documents in Database:
"""
            
            if not documents:
                return header + "\n*No documents currently loaded*"
            
            # Create table format for documents
            table_header = "| # | Document Name | Status |\n|---|---|---|\n"
            table_rows = ""
            
            for i, doc in enumerate(documents, 1):
                # Clean filename for display
                clean_name = doc.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
                if len(clean_name) > 40:
                    clean_name = clean_name[:40] + "..."
                
                status = "✅ Processed"
                table_rows += f"| {i} | {clean_name} | {status} |\n"
            
            return header + table_header + table_rows
            
        except Exception as e:
            return f"❌ Database error: {str(e)}"

    def reset_database(self) -> str:
        """Reset database"""
        try:
            success = self.vector_store.reset_collection()
            if success:
                self.is_ready_for_chat = False
                return "✅ **Database Reset Complete**\n\nAll chunks removed. Upload new documents to rebuild the database.\n\n**Note:** You may need to restart Talk2Doc for optimal performance."
            return "❌ Failed to reset database"
        except Exception as e:
            return f"❌ Reset error: {str(e)}"

    def create_interface(self):
        """Create streamlined interface"""
        
        with gr.Blocks(
            title="💬 Talk2Doc",
            theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="blue")
        ) as demo:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; background: linear-gradient(45deg, #1ABC9C, #3498DB); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                        font-size: 2.5rem; font-weight: bold; margin: 2rem 0;">
                💬 Talk2Doc
            </div>
            <div style="text-align: center; color: #666; margin-bottom: 30px;">
                🤝 Your Intelligent Document Companion
            </div>
            """)
            
            # API Status
            with gr.Row():
                api_status = gr.Textbox(label="🔑 API Status", interactive=False)
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            # Processing Status
            status_display = gr.Textbox(label="⚡ Status", interactive=False, value="💭 Ready")
            
            # Main Tabs
            with gr.Tabs():
                
                # Upload Tab
                with gr.TabItem("📤 Upload Document"):
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="Upload Document (PDF, DOCX, TXT)",
                                file_types=['.pdf', '.docx', '.txt']
                            )
                            upload_btn = gr.Button("🚀 Upload", variant="primary", size="lg")
                        with gr.Column():
                            upload_status = gr.Markdown()
                    
                    with gr.Row():
                        current_file = gr.Textbox(label="📄 Current File", interactive=False)
                        file_url = gr.Textbox(label="🔗 File URL", interactive=False)
                
                # Summary Tab
                with gr.TabItem("📋 Summary"):
                    with gr.Row():
                        with gr.Column():
                            length_radio = gr.Radio(
                                choices=["Short", "Standard", "Detailed"], 
                                value="Standard", 
                                label="📏 Summary Length"
                            )
                            summarize_btn = gr.Button(
                                "✨ Generate Summary", 
                                variant="primary", 
                                size="lg"
                            )
                        with gr.Column():
                            gr.HTML("""
                            <div style="background: rgba(26, 188, 156, 0.1); padding: 15px; border-radius: 10px;">
                                <h4>📋 Summary Options</h4>
                                <ul style="margin: 10px 0; color: #666;">
                                    <li><strong>Short:</strong> Quick overview</li>
                                    <li><strong>Standard:</strong> Balanced summary</li>
                                    <li><strong>Detailed:</strong> Comprehensive analysis</li>
                                </ul>
                            </div>
                            """)
                    
                    summary_output = gr.Markdown(
                        label="📋 Document Summary",
                        elem_classes=["summary-output"],
                        height=300
                    )
                
                # Chat Tab
                with gr.TabItem("💬 Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                height=500,
                                label="💬 Document Assistant",
                                type="messages"
                            )
                            
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    placeholder="Ask anything about your document...",
                                    scale=4,
                                    lines=1,  # Single line enables Enter to submit
                                    max_lines=3  # But allows expansion if needed
                                )
                                send_btn = gr.Button("📤 Send", variant="primary")
                            
                            with gr.Row():
                                clear_btn = gr.Button("🧹 Clear", size="sm")
                                status_btn = gr.Button("📊 Status", size="sm")
                        
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div style="background: linear-gradient(135deg, rgba(26, 188, 156, 0.1), rgba(52, 152, 219, 0.1)); 
                                        border-radius: 15px; padding: 20px;">
                                <h3>💡 Smart Features</h3>
                                <ul style="list-style: none; padding: 0; font-size: 0.9em;">
                                    <li>🔍 <strong>Smart Analysis</strong></li>
                                    <li>📚 <strong>Context Aware</strong></li>
                                    <li>📝 <strong>Dynamic Bullets</strong></li>
                                    <li>⚡ <strong>Auto Processing</strong></li>
                                </ul>
                                
                                <h4>🎯 Try These Questions:</h4>
                                <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin: 10px 0;">
                                    <div style="font-size: 0.85em; color: #A0AEC0;">
                                        <strong>📊 Analysis:</strong><br>
                                        "What's the main theme?"<br>
                                        "What's the moral?"<br><br>
                                        
                                        <strong>📋 Lists:</strong><br>
                                        "Give me 7 key points"<br>
                                        "List important features"<br><br>
                                        
                                        <strong>ℹ️ Facts:</strong><br>
                                        "Who is the author?"<br>
                                        "What's the course fee?"<br><br>
                                        
                                        <strong>📄 Summary:</strong><br>
                                        "Summarize in short"<br>
                                        "What is this about?"
                                    </div>
                                </div>
                                
                                <div style="text-align: center; margin-top: 15px;">
                                    <small style="color: #718096;">✨ Supports any document type</small>
                                </div>
                            </div>
                            """)
                
                # Database Tab
                with gr.TabItem("🗄️ Database"):
                    gr.HTML("""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h3>🗄️ Vector Database Management</h3>
                        <p style="color: #666;">Manage your processed documents and database storage</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        db_info_btn = gr.Button("📊 Database Status", variant="secondary", size="lg")
                        reset_btn = gr.Button("🔄 Reset Database", variant="stop", size="lg")
                    
                    db_output = gr.Markdown(height=400)
                    
                    # Auto-refresh database info when tab loads
                    with gr.Column():
                        gr.HTML("""
                        <div style="background: rgba(255, 193, 7, 0.1); padding: 10px; border-radius: 8px; margin-top: 15px;">
                            <small><strong>💡 Tip:</strong> Use Reset Database to clear all processed documents and start fresh with Talk2Doc.</small>
                        </div>
                        """)
            
            # Event Handlers
            refresh_btn.click(self.validate_api_key, outputs=[api_status])
            upload_btn.click(self.upload_document, inputs=[file_upload], outputs=[upload_status, current_file, file_url])
            summarize_btn.click(self.generate_summary, inputs=[length_radio], outputs=[summary_output])
            
            # Chat events - BOTH send button AND enter key
            send_btn.click(
                fn=self.chat_with_document,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )
            chat_input.submit(
                fn=self.chat_with_document,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )
            
            clear_btn.click(self.clear_chat_history, outputs=[chatbot])
            status_btn.click(self.get_processing_status, outputs=[status_display])
            db_info_btn.click(self.get_collection_info, outputs=[db_output])
            reset_btn.click(self.reset_database, outputs=[db_output])
            
            # Initialize
            demo.load(self.validate_api_key, outputs=[api_status])
        
        return demo

def main():
    """Launch application"""
    try:
        app = RAGDocumentApp()
        demo = app.create_interface()
        
        print("🚀 Starting Talk2Doc...")
        print("✨ Enhanced document processing with intelligent analysis")
        print("💬 Advanced AI-powered chat interface")
        print("⚡ Background processing for seamless experience")
        print("⌨️ Enter key support for chat input")
        
        demo.launch(
            server_port=8080, 
            share=False, 
            server_name="localhost"
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        print(f"❌ Startup failed: {str(e)}")

if __name__ == "__main__":
    main()
