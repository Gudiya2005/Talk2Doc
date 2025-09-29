import os
from dotenv import load_dotenv
import logging

load_dotenv()
# Add these logger lines
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    # Gemini Configuration (replacing OpenAI)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL = "gemini-2.5-flash"  # Best model for documents
    TEMPERATURE = 0.0
    MAX_TOKENS = 1500
    
    # Summary configuration - THIS WAS MISSING!
    SUMMARY_BUDGETS = {
        "short": (20, 75),      # 20-75 words for short summaries
        "standard": (150, 300), # 150-300 words for standard
        "detailed": (400, 800)  # 400-800 words for detailed
    }
    SUMMARY_DEFAULT_MODE = "standard"
    
    # Keep all your existing settings
    CHUNK_SIZE = 1200      # Bigger chunks = fewer embeddings
    CHUNK_OVERLAP = 160    # Less overlap = less processing
    VECTOR_DB_PATH = "data/vectordb"
    COLLECTION_NAME = "documents"
    UPLOAD_FOLDER = "data/docs_upload"
    DOCS_FOLDER = "data/docs"
    SUMMARIES_FOLDER = "data/summaries"
    SERVER_PORT = 8000
    SERVER_HOST = "localhost"
    GRADIO_PORT = 7860
    GRADIO_SHARE = False
    SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt']
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def validate_config(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Please set it in .env file")
    
        # Create all necessary directories with proper error handling
        directories = [
            cls.UPLOAD_FOLDER,
            cls.DOCS_FOLDER, 
            cls.SUMMARIES_FOLDER,
            cls.VECTOR_DB_PATH
        ]
    
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
                raise
    
        return True
