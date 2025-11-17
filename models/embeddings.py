"""
Embedding Models for RAG
Uses Sentence Transformers for document embeddings
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from config.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Handles text embeddings for RAG"""

    def __init__(self, model_name=None):
        """
        Initialize embedding model

        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            self.model_name = model_name or EMBEDDING_MODEL
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            raise Exception(f"Error loading embedding model: {str(e)}")

    def encode_text(self, text):
        """
        Convert text to embedding vector

        Args:
            text: Input text string or list of strings

        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(text, str):
                text = [text]
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            raise Exception(f"Error encoding text: {str(e)}")

    def encode_documents(self, documents):
        """
        Encode multiple documents

        Args:
            documents: List of document texts

        Returns:
            Numpy array of document embeddings
        """
        try:
            embeddings = self.model.encode(documents, 
                                          convert_to_numpy=True,
                                          show_progress_bar=True)
            return embeddings
        except Exception as e:
            raise Exception(f"Error encoding documents: {str(e)}")

    def compute_similarity(self, query_embedding, doc_embeddings):
        """
        Compute cosine similarity between query and documents

        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding vectors

        Returns:
            Similarity scores
        """
        try:
            # Ensure correct shape
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Compute cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

            similarities = np.dot(query_norm, doc_norm.T).flatten()
            return similarities
        except Exception as e:
            raise Exception(f"Error computing similarity: {str(e)}")
