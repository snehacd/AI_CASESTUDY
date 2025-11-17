"""
RAG (Retrieval Augmented Generation) Utilities
Handles document processing, chunking, and retrieval
"""
import os
import PyPDF2
import faiss
import numpy as np
from typing import List, Tuple
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Process and chunk documents for RAG"""

    @staticmethod
    def read_pdf(file_path):
        """
        Read text from PDF file

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    @staticmethod
    def read_txt(file_path):
        """
        Read text from TXT file

        Args:
            file_path: Path to text file

        Returns:
            File contents
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    @staticmethod
    def chunk_text(text, chunk_size=None, overlap=None):
        """
        Split text into overlapping chunks

        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        try:
            chunk_size = chunk_size or CHUNK_SIZE
            overlap = overlap or CHUNK_OVERLAP

            chunks = []
            start = 0
            text_length = len(text)

            while start < text_length:
                end = start + chunk_size
                chunk = text[start:end]

                # Try to end at sentence boundary
                if end < text_length:
                    last_period = chunk.rfind('.')
                    last_newline = chunk.rfind('\n')
                    boundary = max(last_period, last_newline)

                    if boundary > chunk_size * 0.5:  # At least 50% of chunk
                        chunk = chunk[:boundary + 1]
                        end = start + boundary + 1

                chunks.append(chunk.strip())
                start = end - overlap

            return chunks
        except Exception as e:
            raise Exception(f"Error chunking text: {str(e)}")


class VectorStore:
    """FAISS-based vector store for document retrieval"""

    def __init__(self, embedding_model):
        """
        Initialize vector store

        Args:
            embedding_model: Embedding model instance
        """
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.dimension = None

    def add_documents(self, documents: List[str]):
        """
        Add documents to vector store

        Args:
            documents: List of document chunks
        """
        try:
            # Encode documents
            embeddings = self.embedding_model.encode_documents(documents)

            # Initialize FAISS index if not exists
            if self.index is None:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)

            # Add to index
            self.index.add(embeddings.astype('float32'))
            self.documents.extend(documents)

            print(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            raise Exception(f"Error adding documents: {str(e)}")

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for relevant documents

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, distance) tuples
        """
        try:
            if self.index is None or len(self.documents) == 0:
                return []

            # Encode query
            query_embedding = self.embedding_model.encode_text(query)

            # Search
            k = min(k, len(self.documents))
            distances, indices = self.index.search(query_embedding.astype('float32'), k)

            # Return results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(distance)))

            return results
        except Exception as e:
            raise Exception(f"Error searching: {str(e)}")

    def clear(self):
        """Clear all documents from vector store"""
        self.index = None
        self.documents = []
        self.dimension = None
