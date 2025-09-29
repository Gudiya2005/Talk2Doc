import logging
from typing import List, Dict, Any
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self):
        self.config = Config()
        
        # Initialize both APIs for flexibility
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
        
        # LangChain model for advanced features
        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.config.GEMINI_MODEL,
            temperature=self.config.TEMPERATURE,
            google_api_key=self.config.GEMINI_API_KEY
        )
        
        self.conversation_history = []
        self.setup_advanced_chains()

    def setup_advanced_chains(self):
        """Setup specialized LangChain chains for different question types"""
        
        # Character Analysis Chain
        self.character_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a literature expert. Focus on character identification and analysis."),
            ("human", """Based on the document content, identify and analyze the main character.

Context: {context}

Question: {input}

Provide a focused answer about WHO the character is and their role (2-3 sentences max):""")
        ])
        
        # Analytical Questions Chain (Themes, Morals, Significance)
        self.analytical_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing themes, morals, and deeper meanings. Focus on interpretation, not plot summary."),
            ("human", """Analyze the deeper meaning and significance based on the document.

Context: {context}

Question: {input}

Focus on themes, morals, lessons, or significance (not plot details):""")
        ])
        
        # Direct Answer Chain (Facts, Simple Questions)
        self.direct_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise fact-finder. Provide direct, concise answers to factual questions."),
            ("human", """Answer this question directly and concisely based on the document content.

Context: {context}

Question: {input}

Provide a direct, factual answer (1-2 sentences):""")
        ])
        
        # Explanatory Chain (How, Why, Process)
        self.explanatory_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert explainer. Provide clear, detailed explanations of concepts and processes."),
            ("human", """Explain this concept clearly based on the document content.

Context: {context}

Question: {input}

Provide a clear, comprehensive explanation:""")
        ])

        # Create the chains
        self.character_chain = create_stuff_documents_chain(self.langchain_model, self.character_prompt)
        self.analytical_chain = create_stuff_documents_chain(self.langchain_model, self.analytical_prompt)
        self.direct_chain = create_stuff_documents_chain(self.langchain_model, self.direct_prompt)
        self.explanatory_chain = create_stuff_documents_chain(self.langchain_model, self.explanatory_prompt)

    def validate_api_key(self) -> bool:
        """Test Gemini API key validity"""
        try:
            self.model.generate_content("Hello")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False

    def select_appropriate_chain(self, question_type: str, question: str, context_docs: List[Dict]):
        """Select and execute the appropriate LangChain chain based on question type"""
        try:
            # Convert context to LangChain document format
            from langchain_core.documents import Document
            docs = [Document(page_content=chunk['content'], metadata=chunk.get('metadata', {})) 
                   for chunk in context_docs]
            
            # Select appropriate chain
            if question_type == "character_analysis":
                return self.character_chain.invoke({"context": docs, "input": question})
            elif question_type == "analytical":
                return self.analytical_chain.invoke({"context": docs, "input": question})
            elif question_type == "direct_answer":
                return self.direct_chain.invoke({"context": docs, "input": question})
            elif question_type == "explanatory":
                return self.explanatory_chain.invoke({"context": docs, "input": question})
            else:
                # Fallback to direct chain
                return self.direct_chain.invoke({"context": docs, "input": question})
                
        except Exception as e:
            logger.error(f"LangChain processing error: {str(e)}")
            # Fallback to original method
            return self.generate_response_with_memory_fallback(question, context_docs)

    def generate_response_with_memory_fallback(self, user_message: str, context_chunks: List[Dict]) -> str:
        """Fallback method using original approach"""
        try:
            context = "\n\n".join(chunk['content'] for chunk in context_chunks[:5])
            
            prompt = f"""Answer this question based on the document context:

Question: {user_message}

Context: {context}

Provide a clear, focused answer:"""

            response = self.model.generate_content(prompt)
            return response.text.strip() if response and hasattr(response, 'text') else "Unable to generate response"
            
        except Exception as e:
            logger.error(f"Fallback method error: {str(e)}")
            return f"Error generating response: {str(e)}"

    def generate_response_with_memory(self, user_message: str, context_chunks: List[Dict], filename: str, question_type: str = "comprehensive") -> str:
        """Enhanced response generation with LangChain integration"""
        try:
            # Try LangChain approach first for better responses
            if question_type in ["character_analysis", "analytical", "direct_answer", "explanatory"]:
                response = self.select_appropriate_chain(question_type, user_message, context_chunks)
                if response and len(response.strip()) > 10:
                    self.conversation_history.append({
                        'user': user_message,
                        'assistant': response
                    })
                    return response.strip()
            
            # Fallback to original method
            return self.generate_response_with_memory_original(user_message, context_chunks, filename)
            
        except Exception as e:
            logger.error(f"Error in enhanced response generation: {str(e)}")
            return self.generate_response_with_memory_original(user_message, context_chunks, filename)

    def generate_response_with_memory_original(self, user_message: str, context_chunks: List[Dict], filename: str) -> str:
        """Original response generation method (keeping as fallback)"""
        try:
            # Build document context
            context = "\n\n".join(chunk['content'] for chunk in context_chunks)
            
            # Build conversation context (last 3 exchanges)
            conversation_context = ""
            if self.conversation_history:
                recent = self.conversation_history[-3:]
                for exchange in recent:
                    conversation_context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
            
            # Check for SHORT summary request
            user_msg_lower = user_message.lower()
            is_short_request = (
                "in short" in user_msg_lower or 
                "short summary" in user_msg_lower or
                "summarize in short" in user_msg_lower or
                "brief summary" in user_msg_lower or
                ("summarize" in user_msg_lower and "short" in user_msg_lower)
            )
            
            if is_short_request:
                # Force SHORT response with explicit constraints
                prompt = f"""USER REQUEST: "{user_message}"

INSTRUCTION: Provide EXACTLY 2-3 sentences summarizing this document. DO NOT write more than 3 sentences.

DOCUMENT: {filename}
CONTENT: {context[:1000]}

RESPONSE (Maximum 3 sentences):"""
            else:
                # Regular detailed response
                prompt = f"""You are an intelligent document assistant. Your task is to help users clearly understand the given document.

DOCUMENT: {filename}

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_context}

CURRENT QUESTION: {user_message}

Please provide a helpful, accurate response based on the document context."""

            # Generate response
            response = self.model.generate_content(prompt)
            
            if not response or not hasattr(response, 'text') or not response.text:
                return "I encountered an error generating the response. Please try again."
            
            # Store in conversation history
            self.conversation_history.append({
                'user': user_message,
                'assistant': response.text
            })
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error: {str(e)}"

    def format_bullet_response(self, response_text):
        """Clean up bullet point formatting"""
        if "•" in response_text or "- " in response_text or "* " in response_text:
            lines = response_text.split('\n')
            clean_lines = []
            for line in lines:
                clean_line = line.strip(' "•*')
                if clean_line and not clean_line.startswith('Sources'):
                    if any(line.strip().startswith(x) for x in ['- ', '• ', '* ']):
                        bullet_text = clean_line[2:] if len(clean_line) > 2 else clean_line
                        clean_lines.append('- ' + bullet_text.strip())
                    elif clean_line and not clean_line.startswith('-'):
                        clean_lines.append(clean_line)
            return '\n'.join(clean_lines)
        return response_text

    def generate_summary(self, text: str, mode: str = None) -> str:
        """Generate comprehensive or length-controlled summary of document"""
        try:
            if not text or len(text.strip()) < 10:
                return "Document text missing or too short."

            mode = mode or self.config.SUMMARY_DEFAULT_MODE

            # ULTRA-STRICT short mode with content-aware prompting
            if mode == "short":
                # Detect document type first
                text_sample = text[:1000].lower()
                if any(word in text_sample for word in ['story', 'character', 'narrative', 'plot']):
                    prompt = f"""This is a literary work. Write exactly 2-3 sentences about the main plot/story. No administrative details.

Story content: {text[:3000]}

Plot summary (2-3 sentences):"""
                elif any(word in text_sample for word in ['application', 'apply', 'vacancy', 'eligibility']):
                    prompt = f"""This is an application/recruitment document. Write 2-3 sentences about who can apply and key requirements.

Document: {text[:3000]}

Requirements summary (2-3 sentences):"""
                else:
                    prompt = f"""Write exactly 2-3 sentences summarizing the main points. No unnecessary sections.

Document: {text[:3000]}

Summary (2-3 sentences):"""
                
                # Generate and process response for short mode
                response = self.model.generate_content(prompt)
                if response and hasattr(response, 'text') and response.text:
                    summary = response.text.strip()
                    
                    # Force truncation to first 2-3 sentences only
                    sentences = [s.strip() + '.' for s in summary.split('.') if s.strip()]
                    if len(sentences) > 3:
                        sentences = sentences[:3]
                    
                    short_summary = ' '.join(sentences)
                    
                    # Final word count check - hard truncate if needed  
                    words = short_summary.split()
                    if len(words) > 75:
                        short_summary = ' '.join(words[:75]) + '.'
                    
                    return short_summary
                else:
                    return "No valid response from AI. Try again."
            
            # Regular mode (standard/detailed)
            else:
                low, high = self.config.SUMMARY_BUDGETS.get(mode, self.config.SUMMARY_BUDGETS[self.config.SUMMARY_DEFAULT_MODE])
                
                prompt = f"""Write a {mode} summary of the document within {low}-{high} words.
Requirements:
- Preserve key points and critical facts only.
- Avoid speculation and do not add information not present in the text.
- Use clear paragraphs.
Document content:
{text[:20000]}
Now write the summary:"""

                response = self.model.generate_content(prompt)
                if response and hasattr(response, 'text') and response.text:
                    return response.text.strip()
                else:
                    return "No valid response from AI. Try again."

        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return f"Summary error: {str(e)}"

    def clear_conversation_history(self):
        """Clear conversation memory"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate text embeddings (fallback method)"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [0.1] * 768
