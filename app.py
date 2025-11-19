import streamlit as st
import os
import io
from groq import Groq

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings   # <-- Modern Embeddings API

# --- CONFIGURATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, AttributeError):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)

# You MUST include the PDF inside your repo!
BASE_KNOWLEDGE_PATH = "https://github.com/DBR-7/Legal-Chatbot/blob/main/Indian_penal_code.pdf"


class DualLegalRAG:

    def __init__(self):
        # Modern Embedding model (this replaces HuggingFaceBgeEmbeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )

        # Groq client
        if not GROQ_API_KEY:
            st.error("Missing GROQ_API_KEY")
            st.stop()

        self.client = Groq(api_key=GROQ_API_KEY)

        # Session state for vector DBs
        if "base_db" not in st.session_state:
            st.session_state.base_db = None
        if "user_db" not in st.session_state:
            st.session_state.user_db = None

        # Load IPC on first run
        if st.session_state.base_db is None:
            self.load_base_law()

    # -------------------- BASE LAW -------------------------
    @st.cache_resource
    def load_base_law(self):
        if not os.path.exists(BASE_KNOWLEDGE_PATH):
            st.error("IPC PDF missing in repo!")
            return None

        try:
            with st.spinner("Loading Indian Penal Code..."):
                docs = PyPDFLoader(BASE_KNOWLEDGE_PATH).load()

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            ).split_documents(docs)

            base_db = FAISS.from_documents(chunks, self.embeddings)
            st.session_state.base_db = base_db
            st.success(f"IPC Loaded ({len(chunks)} chunks)")
            return base_db

        except Exception as e:
            st.error(f"IPC Load Error: {e}")
            return None

    # -------------------- USER UPLOAD -------------------------
    def process_user_upload(self, uploaded_file):
        if uploaded_file is None:
            st.session_state.user_db = None
            return "No file selected."

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            docs = PyPDFLoader(temp_path).load()
            os.unlink(temp_path)

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            ).split_documents(docs)

            st.session_state.user_db = FAISS.from_documents(chunks, self.embeddings)
            return f"Loaded {len(chunks)} chunks from {uploaded_file.name}"

        except Exception as e:
            st.session_state.user_db = None
            return f"Error: {e}"

    # -------------------- RETRIEVAL -------------------------
    def retrieve(self, query):
        context_parts = []

        # IPC search
        if st.session_state.base_db:
            try:
                results = st.session_state.base_db.similarity_search(query, k=3)
                if results:
                    text = "\n".join(doc.page_content for doc in results)
                    context_parts.append(f"--- IPC ---\n{text}")
            except:
                pass

        # User doc search
        if st.session_state.user_db:
            try:
                results = st.session_state.user_db.similarity_search(query, k=3)
                if results:
                    text = "\n".join(doc.page_content for doc in results)
                    context_parts.append(f"--- USER DOCUMENT ---\n{text}")
            except:
                pass

        return "\n\n".join(context_parts)

    # -------------------- CHAT -------------------------
    def chat(self, message):
        context = self.retrieve(message)

        if context:
            system_msg = (
                "You are a Senior Indian Legal Advisor. "
                "Answer ONLY using the context below."
            )
            user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {message}"
            st.expander("Context Used").markdown(context)
        else:
            system_msg = "You are a legal expert. No documents available."
            user_msg = message

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {e}"


# ============================
# STREAMLIT UI
# ============================

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = DualLegalRAG()

rag = st.session_state.rag_engine

st.set_page_config(page_title="Dual RAG Legal Assistant", layout="centered")
st.title("⚖️ Dual RAG Legal Assistant")

with st.sidebar:
    st.header("Upload Case PDF")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Process"):
        st.info(rag.process_user_upload(uploaded))

# Chat Section
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.chat(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
