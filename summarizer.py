import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from document_processor import DocumentProcessor
from llm_handler import LLMHandler
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentSummarizer:
    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor()
        self.llm_handler = LLMHandler()

    def extract_text_from_document(self, document_path: str) -> str:
        """Extract text content from document - FIXES YOUR ERROR!"""
        try:
            doc_info = self.doc_processor.get_document_info(document_path)
            text_content = doc_info.get('text_content', '')
            
            if not text_content and 'content' in doc_info:
                text_content = doc_info['content']
                
            if not text_content:
                if document_path.endswith('.txt'):
                    with open(document_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                else:
                    raise Exception('No text content found')
                    
            return text_content
        except Exception as e:
            raise Exception(f'Failed to extract text: {str(e)}')

    def generate_summary(self, document_path: str, length_mode: str = "standard") -> dict:
        """Generate summary with length control"""
        try:
            # Check if file exists first
            if not os.path.exists(document_path):
                logger.error(f"Document not found: {document_path}")
                return {"error": "Document file not found"}
            
            # Use your own extract method instead of doc_processor.extract_text_from_file
            content = self.extract_text_from_document(document_path)
            
            if not content:
                return {"error": "Could not extract text from document"}
            
            logger.info(f"Extracted {len(content)} characters for summary")
            
            # Generate summary using LLM handler with length mode
            summary_text = self.llm_handler.generate_summary(content, mode=length_mode)
            
            if not summary_text:
                return {"error": "LLM returned empty summary"}
            
            return {
                "summary": summary_text,
                "length_mode": length_mode,
                "document": os.path.basename(document_path)
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return {"error": str(e)}


    def _save_summary(self, summary_data: Dict[str, Any]) -> str:
        """Save summary to file"""
        os.makedirs(self.config.SUMMARIES_FOLDER, exist_ok=True)
        filename = summary_data.get('filename', f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        summary_filename = f"{Path(filename).stem}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path = os.path.join(self.config.SUMMARIES_FOLDER, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f'Summary saved: {summary_path}')
        return summary_path

    def load_summary(self, summary_path: str) -> Dict[str, Any]:
        """Load summary from file"""
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def delete_summary(self, summary_path: str) -> bool:
        """Delete summary file"""
        try:
            os.remove(summary_path)
            logger.info(f'Summary deleted: {summary_path}')
            return True
        except Exception as e:
            logger.error(f'Error deleting summary: {str(e)}')
            return False

    def list_summaries(self) -> List[Dict[str, Any]]:
        """List all saved summaries"""
        summaries = []
        try:
            if not os.path.exists(self.config.SUMMARIES_FOLDER):
                return summaries
                
            files = os.listdir(self.config.SUMMARIES_FOLDER)
            for file in files:
                if file.endswith('.json'):
                    path = os.path.join(self.config.SUMMARIES_FOLDER, file)
                    try:
                        data = self.load_summary(path)
                        summaries.append({
                            'filename': data.get('filename', 'Unknown'),
                            'created': data.get('generated_at', 'Unknown'),
                            'path': path,
                        })
                    except:
                        continue
                        
            summaries.sort(key=lambda x: x['created'], reverse=True)
            return summaries
        except Exception as e:
            logger.error(f'Error listing summaries: {str(e)}')
            return []
