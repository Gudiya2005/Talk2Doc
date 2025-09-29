import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import logging
import hashlib

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        from config import Config
        self.config = Config()
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create persistent client
        self.client = chromadb.PersistentClient(path=self.config.VECTOR_DB_PATH)
        
        # Load or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.config.COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            logger.info("Loaded existing collection")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 128,
                    "hnsw:M": 16
                }
            )
            logger.info("Created new collection")

    def generate_embeddings(self, text: str):
        """Generate embeddings for text"""
        return self.embedding_function([text])[0]

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks with batch processing"""
        try:
            print(f"🔍 Adding {len(chunks)} chunks to vector database")
            
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_id = self.generate_chunk_id(chunk)
                    documents.append(chunk['content'])
                    metadatas.append(chunk['metadata'])
                    ids.append(chunk_id)
                    
                    # Progress update
                    if (i + 1) % 16 == 0 or i == len(chunks) - 1:
                        progress = (i + 1) * 100 // len(chunks)
                        print(f"🔍 Processed {progress}% ({i + 1}/{len(chunks)} chunks)")
                        
                except Exception as e:
                    print(f"⚠️ Skipping chunk {i}: {e}")
                    continue
            
            # Bulk insert with automatic embedding generation
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Successfully added {len(documents)} chunks to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def generate_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """Generate unique ID for chunk"""
        content_sample = chunk['content'][:100]
        filename = chunk['metadata'].get('filename', 'unknown')
        chunk_num = chunk['metadata'].get('chunk_id', 0)
        
        unique_string = f"{filename}_{chunk_num}_{content_sample}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for semantically similar chunks"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, 50),
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for i in range(len(docs)):
                    # Filter by relevance threshold
                    if distances[i] < 2.0:
                        formatted_results.append({
                            'content': docs[i],
                            'metadata': metas[i],
                            'distance': distances[i],
                            'similarity': max(0, 1 - distances[i]/2)
                        })
            
            # Sort by relevance (lowest distance = highest relevance)
            formatted_results.sort(key=lambda x: x['distance'])
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {str(e)}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get database collection information"""
        try:
            total_chunks = self.collection.count()
            
            # Get document names from metadata
            documents = set()
            if total_chunks > 0:
                sample = self.collection.get(
                    limit=min(total_chunks, 100),
                    include=["metadatas"]
                )
                
                if sample['metadatas']:
                    for metadata in sample['metadatas']:
                        if metadata and 'filename' in metadata:
                            documents.add(metadata['filename'])
            
            return {
                'total_chunks': total_chunks,
                'total_documents': len(documents),
                'documents': list(documents)
            }
            
        except Exception as e:
            return {'error': str(e)}

    def reset_collection(self) -> bool:
        """Reset vector database by deleting and recreating collection"""
        try:
            # Delete existing collection
            self.client.delete_collection(self.config.COLLECTION_NAME)
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 128,
                    "hnsw:M": 16
                }
            )
            
            logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False
