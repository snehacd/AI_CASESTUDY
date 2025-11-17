"""
Configuration Template
Copy this to config.py and add your actual API keys
"""
import os

# LLM API Keys - Replace with your actual keys
OPENAI_API_KEY = "sk-your-openai-api-key-here"
GROQ_API_KEY = "gsk_your-groq-api-key-here"
GEMINI_API_KEY = "your-gemini-api-key-here"

# Web Search API Key (SerpAPI)
SERPAPI_KEY = "your-serpapi-key-here"

# Or use environment variables (recommended for production)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Model Settings
DEFAULT_LLM_PROVIDER = "openai"  # Options: openai, groq, gemini
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS = 1000

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Response Mode
DEFAULT_RESPONSE_MODE = "concise"  # Options: concise, detailed
