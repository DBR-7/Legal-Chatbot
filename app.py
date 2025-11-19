import streamlit as st
import os
import io
from groq import Groq

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceBgeEmbeddings

# --- CONFIGURATION ---
# IMPORTANT: For deployment, ensure GROQ_API_KEY is set as a
# Streamlit secret named 'GROQ_API_KEY'.
# Get it from secrets, or environment variable for local testing.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, AttributeError):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# Path to your STATIC "Behind the Scenes" Legal Knowledge
# On a cloud environment like Streamlit, you should typically
# upload this file to the repo and reference it directly.
BASE_KNOWLEDGE_PATH = "https://github.com/DBR-7/Legal-Chatbot/blob/main/Indian_penal_code.pdf"
# You MUST include a copy of 'Indian_penal_code.pdf' in the same
# directory as this script in your GitHub repository.

class DualLegalRAG:
    """
    RAG engine that integrates context from a static 'Base Law' document
    and a dynamically uploaded 'User Document'.
    """
    def __init__(self):
        # 1. Embeddings Model Initialization
        # Use an appropriate model name and ensure the embedding library is installed.
        # Streamlit cloud runs on Linux, so BGE/FastEmbed should work,
        # but we stick to BGE for consistent results with the original request.
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            # We don't specify device here; let HuggingFace handle it.
            # model_kwargs={"device": "cpu"}
        )

        # 2. Groq Client Initialization
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found. Please set it in Streamlit secrets or environment variables.")
            st.stop()
        self.client = Groq(api_key=GROQ_API_KEY)
        
        # 3. Vector Stores (in Streamlit session state)
        # We use st.session_state to persist these across user interactions.
        if "base_db" not in st.session_state:
            st.session_state.base_db = None
        if "user_db" not in st.session_state:
            st.session_state.user_db = None
            
        # 4. Load the base PDF immediately on first run
        if st.session_state.base_db is None:
            self.load_base_law()

    # ----------------------- LOAD BASE LAW -----------------------
    @st.cache_resource
    def load_base_law(self):
        """Loads and processes the static base legal document."""
        if not os.path.exists(BASE_KNOWLEDGE_PATH):
            st.error(f"❌ ERROR: Base Law file not found at: {BASE_KNOWLEDGE_PATH}")
            st.info("Please ensure 'Indian_penal_code.pdf' is in the repository.")
            return None

        try:
            with st.spinner(f"📘 Loading Base Law from: {BASE_KNOWLEDGE_PATH}..."):
                loader = PyPDFLoader(BASE_KNOWLEDGE_PATH)
                docs = loader.load()

                if not docs or len(docs[0].page_content.strip()) < 10:
                    st.error("❌ ERROR: Base PDF seems empty or is a scanned image.")
                    return None

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100
                ).split_documents(docs)

                base_db = FAISS.from_documents(chunks, self.embeddings)
                st.session_state.base_db = base_db
                st.success(f"✅ Base Law Loaded successfully ({len(chunks)} chunks)")
                return base_db

        except Exception as e:
            st.error(f"❌ CRITICAL ERROR loading base PDF: {e}")
            return None

    # ----------------------- USER PDF ----------------------------
    def process_user_upload(self, uploaded_file):
        """Processes the PDF uploaded by the user."""
        if uploaded_file is None:
            st.session_state.user_db = None
            return "⚠ No file selected."

        try:
            # Use tempfile to write the uploaded file to disk
            # Streamlit UploadedFile object behaves like a file-like object (io.BytesIO)
            # We need to save it to a temporary path for PyPDFLoader
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            with st.spinner(f"📂 Processing User File: {uploaded_file.name}..."):
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
            
            # Clean up the temp file
            os.unlink(tmp_file_path)

            if not docs:
                st.session_state.user_db = None
                return "❌ Error: Document is empty."

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            ).split_documents(docs)

            st.session_state.user_db = FAISS.from_documents(chunks, self.embeddings)
            return f"✅ Success! Loaded **{len(chunks)}** chunks from **{uploaded_file.name}**."

        except Exception as e:
            st.session_state.user_db = None
            return f"❌ Error processing file: {str(e)}"

    # ----------------------- RETRIEVAL ---------------------------
    def retrieve(self, query):
        """Retrieves context from both base and user vector stores."""
        context_parts = []

        # 1. Search Base Law
        if st.session_state.base_db:
            try:
                base_results = st.session_state.base_db.similarity_search(query, k=3)
                if base_results:
                    text = "\n".join([doc.page_content for doc in base_results])
                    context_parts.append(
                        f"--- OFFICIAL INDIAN PENAL CODE ---\n{text}")
            except Exception as e:
                print(f"Base search error: {e}")

        # 2. Search User Document
        if st.session_state.user_db:
            try:
                user_results = st.session_state.user_db.similarity_search(query, k=3)
                if user_results:
                    text = "\n".join([doc.page_content for doc in user_results])
                    context_parts.append(
                        f"--- USER UPLOADED EVIDENCE/CONTRACT ---\n{text}")
            except Exception as e:
                print(f"User search error: {e}")

        return "\n\n".join(context_parts)

    # ----------------------- MAIN CHAT ---------------------------
    def chat(self, message):
        """Generates a response using Groq, powered by RAG context."""
        context = self.retrieve(message)

        # Fallback if no documents work
        if not context:
            system_msg = "You are a legal expert. Answer based on general knowledge as no documents are loaded."
            user_msg = message
            st.info("No documents are currently loaded. Answering based on general knowledge.")
        else:
            system_msg = "You are a Senior Indian Legal Advisor. Answer strictly based on the CONTEXT provided. Do not invent information."
            user_msg = f"CONTEXT:\n{context}\n\nUSER QUESTION: {message}"
            st.expander("🔍 **Context Used**").markdown(context) # Display the context for debugging/transparency

        try:
            # Use a slightly smaller, faster model if Llama 3 is available
            model_name = "llama3-8b-8192" 
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {str(e)}"

# ====================================================
# STREAMLIT UI LAYOUT
# ====================================================

# Initialize the RAG engine (only runs once thanks to st.cache_resource)
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = DualLegalRAG()
rag_engine = st.session_state.rag_engine


st.set_page_config(page_title="Dual RAG Legal Assistant", layout="centered")
st.title("⚖️ Dual RAG Legal Assistant")
st.markdown(
    """
    This system combines two knowledge bases to provide comprehensive legal advice:
    1. **Base Law:** The static Indian Penal Code (IPC).
    2. **User Case:** Your dynamically uploaded contract, FIR, or evidence document.
    """
)

# --- Sidebar for Upload ---
with st.sidebar:
    st.header("Upload User Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF of your contract, FIR, or case document.",
        type=["pdf"],
        help="This document will be used as secondary context."
    )
    
    # Button to process the file and store the user_db in session state
    if st.button("Process Document"):
        with st.status("Processing file..."):
            status_message = rag_engine.process_user_upload(uploaded_file)
        st.info(status_message)
        
    st.markdown("---")
    st.caption("Base Law: Indian Penal Code (Loaded on startup)")
    
# --- Main Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a legal question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_engine.chat(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history

    st.session_state.messages.append({"role": "assistant", "content": response})
