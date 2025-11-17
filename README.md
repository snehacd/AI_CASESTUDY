# AI Knowledge Assistant - Intelligent Chatbot with RAG and Web Search

A powerful Streamlit-based chatbot application that combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and real-time web search capabilities.

## 🌟 Features

### Core Features
- **Multiple LLM Support**: Integration with OpenAI, Groq, and Google Gemini
- **RAG Integration**: Upload and query local documents (PDF, TXT)
- **Web Search**: Real-time web search when LLM lacks knowledge
- **Response Modes**: Switch between Concise and Detailed responses
- **Persistent Chat**: Maintains conversation history during session

### Technical Highlights
- Vector embeddings using Sentence Transformers
- FAISS-based vector search for document retrieval
- Modular architecture for easy extension
- Error handling with try-except blocks
- Session state management for seamless UX

## 📁 Project Structure

```
project/
├── config/
│   └── config.py          # API keys and settings
├── models/
│   ├── llm.py             # LLM models (OpenAI/Groq/Gemini)
│   └── embeddings.py      # RAG embedding models
├── utils/
│   ├── rag_utils.py       # Document processing and vector store
│   └── web_search.py      # Web search functionality
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

## Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
Edit `config/config.py` and add your API keys:

```python
# Option 1: Direct assignment (not recommended for production)
OPENAI_API_KEY = "your-openai-api-key"
GROQ_API_KEY = "your-groq-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
SERPAPI_KEY = "your-serpapi-key"

# Option 2: Use environment variables (recommended)
# Set environment variables in your system or create a .env file
```

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Groq: https://console.groq.com/keys
- Google Gemini: https://makersuite.google.com/app/apikey
- SerpAPI: https://serpapi.com/manage-api-key

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Chatbot

1. **Load Models**: Click "Load Models" in the sidebar to initialize the LLM
2. **Upload Documents** (Optional): Upload PDF or TXT files for RAG
3. **Enable Features**: Check "Enable RAG" or "Enable Web Search" as needed
4. **Select Response Mode**: Choose between Concise or Detailed responses
5. **Chat**: Type your question in the chat input and press Enter

### Example Use Cases

#### 1. Document Q&A (RAG)
- Upload your company policies, research papers, or documentation
- Enable RAG
- Ask questions about the uploaded documents

#### 2. Current Information (Web Search)
- Enable Web Search
- Ask about recent events, news, or current statistics
- Get answers with source links

#### 3. Hybrid Mode (RAG + Web Search)
- Enable both RAG and Web Search
- Get comprehensive answers from both your documents and the web

## 🔧 Configuration Options

### LLM Settings
- **Provider**: OpenAI, Groq, or Gemini
- **Model**: Select from available models per provider
- **Temperature**: Control response randomness (0-1)

### Response Modes
- **Concise**: Short, summarized replies
- **Detailed**: Expanded, in-depth responses

### RAG Settings (in config.py)
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
CHUNK_SIZE = 500                       # Document chunk size
CHUNK_OVERLAP = 50                     # Overlap between chunks
```

## 🌐 Deployment to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

**Important**: Create a `.gitignore` file:
```
venv/
__pycache__/
*.pyc
.env
temp_*
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Add secrets in "Advanced settings":
   ```
   OPENAI_API_KEY = "your-key"
   GROQ_API_KEY = "your-key"
   GEMINI_API_KEY = "your-key"
   SERPAPI_KEY = "your-key"
   ```
7. Click "Deploy"

## 🏗️ Architecture

### RAG Pipeline
1. **Document Processing**: PDF/TXT files → Text extraction
2. **Chunking**: Text → Overlapping chunks
3. **Embedding**: Chunks → Vector embeddings
4. **Indexing**: Embeddings → FAISS index
5. **Retrieval**: Query → Similar chunks
6. **Generation**: LLM + Context → Response

### Web Search Pipeline
1. User query → SerpAPI
2. Search results → Formatted context
3. Context + Query → LLM
4. LLM → Enhanced response with sources

## 🛠️ Customization

### Adding New LLM Providers
Edit `models/llm.py` and add your provider in the `LLMModel` class.

### Changing Embedding Models
Edit `config/config.py`:
```python
EMBEDDING_MODEL = "your-preferred-model"
```

Popular options:
- `all-MiniLM-L6-v2` (default, fast)
- `all-mpnet-base-v2` (higher quality)
- `multi-qa-mpnet-base-dot-v1` (Q&A optimized)

### Modifying Chunk Size
Edit `config/config.py`:
```python
CHUNK_SIZE = 1000      # Larger chunks = more context
CHUNK_OVERLAP = 100    # Larger overlap = better continuity
```

## 🐛 Troubleshooting

### Common Issues

**1. Module Not Found Error**
```bash
pip install -r requirements.txt
```

**2. API Key Error**
- Check that API keys are correctly set in `config/config.py`
- Ensure no extra spaces or quotes

**3. FAISS Installation Issues on Windows**
```bash
pip install faiss-cpu --no-cache
```

**4. Memory Error with Large Documents**
- Reduce `CHUNK_SIZE` in config
- Process documents in smaller batches

**5. Streamlit Cloud Deployment Issues**
- Ensure all secrets are added in Streamlit Cloud settings
- Check that requirements.txt has all dependencies
- Verify Python version compatibility (3.8+)

## 📊 Performance Tips

1. **Faster Embeddings**: Use smaller models like `all-MiniLM-L6-v2`
2. **Better Quality**: Use `all-mpnet-base-v2` or `multi-qa-mpnet-base-dot-v1`
3. **Reduce API Costs**: Use Groq (free) or lower temperature settings
4. **Optimize RAG**: Adjust chunk size and number of retrieved documents (k parameter)

## 🔒 Security Best Practices

1. **Never commit API keys** to GitHub
2. Use **environment variables** for production
3. Add **config/config.py** to `.gitignore`
4. Use Streamlit Cloud **secrets management**
5. Implement **rate limiting** for production apps

## 📝 Development Guidelines

### Code Standards
- Use **try-except blocks** for error handling
- Add **docstrings** to all functions
- Follow **PEP 8** style guide
- Keep functions **modular and reusable**

### Testing
Test each component individually:
```bash
# Test embeddings
python -c "from models.embeddings import EmbeddingModel; model = EmbeddingModel()"

# Test LLM
python -c "from models.llm import LLMModel; model = LLMModel('openai')"

# Test web search
python -c "from utils.web_search import WebSearch; ws = WebSearch(); print(ws.search('AI'))"
```

## 🎓 Use Case Ideas

1. **Customer Support Bot**: Upload company docs and policies
2. **Research Assistant**: Combine PDFs with web search
3. **Educational Tutor**: Upload course materials
4. **Legal Assistant**: Query legal documents
5. **Medical Information**: Search medical literature
6. **Code Helper**: Upload documentation for libraries

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Streamlit for the amazing framework
- Hugging Face for Sentence Transformers
- OpenAI, Groq, and Google for LLM APIs
- FAISS for vector search capabilities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for the NeoStats AI Engineer Case Study**

openAI key:
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

groq key:

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

Gemini key: 
AIzaSyABeU1Gnn9ZRxPGlc7rh4tIKr5DNAUykTg

serai

59a33fdf87023374f3379829dee2cb209f0a4636c700ac0013db82ed8cc617c4#   A I _ C A S E S T U D Y  
 