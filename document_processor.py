import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import pypdf
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    def save_uploaded_file(self, uploaded_file, filename: str) -> str:
        """Save uploaded file to upload directory"""
        upload_folder = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'data', 'docs_upload'
        ))
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)

        with open(file_path, 'wb') as f:
            if hasattr(uploaded_file, 'read'):
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)
                f.write(uploaded_file.read())
            elif hasattr(uploaded_file, 'name') and os.path.exists(uploaded_file.name):
                with open(uploaded_file.name, 'rb') as source:
                    f.write(source.read())
            else:
                f.write(uploaded_file)

        logger.info(f"File saved: {file_path}")
        return file_path

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i + 1} ---\n{page_text}\n"

        if not text.strip():
            raise ValueError("No text extracted from PDF")

        logger.info(f"Extracted {len(text)} chars from PDF")
        return text

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())

        if not text.strip():
            raise ValueError("No text extracted from DOCX")

        logger.info(f"Extracted {len(text)} chars from DOCX")
        return text

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()

        if not text.strip():
            raise ValueError("Text file is empty")

        logger.info(f"Extracted {len(text)} chars from TXT")
        return text

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from any supported file"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def chunk_document(self, text: str, filename: str, file_path: str = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        chunk_list = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                chunk_list.append({
                    'content': chunk,
                    'metadata': {
                        'filename': filename,
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'source': file_path or filename
                    }
                })
        
        logger.info(f"Created {len(chunk_list)} chunks for {filename}")
        return chunk_list

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Complete document processing pipeline"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) > self.config.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Max: {self.config.MAX_FILE_SIZE} bytes")

        filename = os.path.basename(file_path)
        logger.info(f"Processing document: {filename}")

        # Extract and validate text
        text = self.extract_text_from_file(file_path)
        if len(text.strip()) < 50:
            raise ValueError("Extracted text too short. Check document content.")

        # Create chunks
        chunks = self.chunk_document(text, filename, file_path)
        if not chunks:
            raise ValueError("No valid chunks created")

        logger.info(f"Successfully processed {filename}: {len(chunks)} chunks")
        return chunks

    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get document information and metadata"""
        try:
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_ext = Path(file_path).suffix.lower()

            # Extract text content for additional info
            try:
                text = self.extract_text_from_file(file_path)
                char_count = len(text)
                word_count = len(text.split())
                text_content = text  # Add this for summarizer compatibility
            except Exception:
                char_count = 0
                word_count = 0
                text_content = ""

            return {
                'filename': filename,
                'file_size': file_size,
                'file_extension': file_ext,
                'character_count': char_count,
                'word_count': word_count,
                'text_content': text_content,  # ✅ This fixes the summarizer issue!
                'supported': file_ext in self.config.SUPPORTED_FORMATS
            }

        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return {
                'filename': os.path.basename(file_path) if file_path else 'Unknown',
                'error': str(e)
            }
