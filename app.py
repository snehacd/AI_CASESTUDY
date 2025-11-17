"""
Intelligent Chatbot with RAG and Web Search
Built with Streamlit, LLMs, and Vector Search
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import *
from models.llm import LLMModel
from models.embeddings import EmbeddingModel
from utils.rag_utils import DocumentProcessor, VectorStore
from utils.web_search import WebSearch


# Page configuration
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = None

    if "web_search" not in st.session_state:
        st.session_state.web_search = None

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


def load_models(provider, model_name, temperature):
    """Load LLM and embedding models"""
    try:
        # Initialize embedding model if not exists
        if st.session_state.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = EmbeddingModel()
                st.session_state.vector_store = VectorStore(st.session_state.embedding_model)

        # Initialize LLM model
        with st.spinner(f"Loading {provider} model..."):
            st.session_state.llm_model = LLMModel(
                provider=provider,
                model=model_name,
                temperature=temperature
            )

        # Initialize web search
        st.session_state.web_search = WebSearch()

        st.success(f"✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")


def process_uploaded_files(uploaded_files):
    """Process uploaded documents for RAG"""
    try:
        all_chunks = []

        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            # Save temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Process based on file type
            if file_extension == 'pdf':
                text = DocumentProcessor.read_pdf(temp_path)
            elif file_extension == 'txt':
                text = DocumentProcessor.read_txt(temp_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            # Chunk text
            chunks = DocumentProcessor.chunk_text(text)
            all_chunks.extend(chunks)

            # Clean up
            os.remove(temp_path)

        # Add to vector store
        if all_chunks:
            st.session_state.vector_store.add_documents(all_chunks)
            st.session_state.documents_loaded = True
            st.success(f"✅ Processed {len(all_chunks)} document chunks!")

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")


def generate_response(prompt, use_rag, use_web_search, response_mode):
    """Generate chatbot response"""
    try:
        context = ""

        # RAG retrieval
        if use_rag and st.session_state.documents_loaded:
            results = st.session_state.vector_store.search(prompt, k=3)
            if results:
                context += "Relevant Information from Documents:\n\n"
                for doc, distance in results:
                    context += f"{doc}\n\n"

        # Web search
        if use_web_search:
            search_results = st.session_state.web_search.search(prompt, num_results=3)
            web_context = st.session_state.web_search.format_results(search_results)
            context += web_context

        # Adjust prompt based on response mode
        if response_mode == "Concise":
            enhanced_prompt = f"{prompt}\n\nProvide a brief, concise answer."
        else:
            enhanced_prompt = f"{prompt}\n\nProvide a detailed, comprehensive answer."

        # Generate response
        response = st.session_state.llm_model.generate_response(
            enhanced_prompt,
            context=context if context else None
        )

        return response

    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    """Main application"""

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🤖 AI Knowledge Assistant</div>', 
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Settings
        st.subheader("LLM Settings")
        provider = st.selectbox(
            "Select Provider",
            ["openai", "groq", "gemini"],
            index=0
        )

        model_options = {
            "openai": ["gpt-3.5-turbo", "gpt-4"],
            "groq": ["llama3-70b-8192", "mixtral-8x7b-32768"],
            "gemini": ["gemini-pro", "gemini-1.5-pro"]
        }

        model_name = st.selectbox(
            "Select Model",
            model_options[provider]
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )

        if st.button("Load Models"):
            load_models(provider, model_name, temperature)

        st.markdown("---")

        # Response Mode
        st.subheader("Response Mode")
        response_mode = st.radio(
            "Select response style:",
            ["Concise", "Detailed"]
        )

        st.markdown("---")

        # Features
        st.subheader("Features")
        use_rag = st.checkbox("Enable RAG (Document Search)", value=False)
        use_web_search = st.checkbox("Enable Web Search", value=False)

        st.markdown("---")

        # Document Upload
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents"):
            if st.session_state.embedding_model is None:
                st.error("Please load models first!")
            else:
                process_uploaded_files(uploaded_files)

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        # Clear documents
        if st.button("🗑️ Clear Documents"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear()
                st.session_state.documents_loaded = False
                st.success("Documents cleared!")

    # Main chat interface
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Check if models are loaded
        if st.session_state.llm_model is None:
            st.error("⚠️ Please load models first from the sidebar!")
            return

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    prompt,
                    use_rag,
                    use_web_search,
                    response_mode
                )
            st.markdown(response)

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
