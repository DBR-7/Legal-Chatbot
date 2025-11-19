import streamlit as st
import os
import tempfile
from groq import Groq

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =======================================
#              CONFIG
# =======================================
BASE_KNOWLEDGE_PATH = "Indian_penal_code.pdf"

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# =======================================
#   CACHED BASE LAW LOADER (TOP-LEVEL)
# =======================================
@st.cache_resource
def load_base_law(embedding_model, pdf_path):
    """
    Loads the Indian Penal Code (IPC) PDF into a FAISS vector DB.
    IMPORTANT: This must NOT be a class method.
    """

    if not os.path.exists(pdf_path):
        st.error(f"Base PDF missing in repo: {pdf_path}")
        return None

    with st.spinner("Loading Indian Penal Code…"):
        docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embedding_model)

    st.success(f"IPC Loaded Successfully ({len(chunks)} chunks)")
    return db


# =======================================
#           DUAL RAG ENGINE
# =======================================
class DualLegalRAG:

    def __init__(self):
        # Embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )

        # Groq client
        if not GROQ_API_KEY:
            st.error("Missing GROQ_API_KEY")
            st.stop()

        self.client = Groq(api_key=GROQ_API_KEY)

        # Session vector DBs
        if "base_db" not in st.session_state:
            st.session_state.base_db = None
        if "user_db" not in st.session_state:
            st.session_state.user_db = None

        # Load IPC only once (cached)
        if st.session_state.base_db is None:
            st.session_state.base_db = load_base_law(
                self.embeddings,
                BASE_KNOWLEDGE_PATH
            )

    # ---------------------------
    #      USER UPLOADS
    # ---------------------------
    def process_user_upload(self, uploaded_file):
        if uploaded_file is None:
            st.session_state.user_db = None
            return "No file selected."

        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            docs = PyPDFLoader(temp_path).load()
            os.unlink(temp_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)

            st.session_state.user_db = FAISS.from_documents(
                chunks, self.embeddings
            )

            return f"Loaded {len(chunks)} chunks from {uploaded_file.name}"

        except Exception as e:
            st.session_state.user_db = None
            return f"Error loading PDF: {e}"

    # ---------------------------
    #         RETRIEVAL
    # ---------------------------
    def retrieve(self, query):
        context_parts = []

        # --- BASE IPC ---
        if st.session_state.base_db:
            try:
                ipc_results = st.session_state.base_db.similarity_search(query, k=3)
                text = "\n".join(doc.page_content for doc in ipc_results)
                context_parts.append(f"--- IPC ---\n{text}")
            except:
                pass

        # --- USER DOC ---
        if st.session_state.user_db:
            try:
                usr_results = st.session_state.user_db.similarity_search(query, k=3)
                text = "\n".join(doc.page_content for doc in usr_results)
                context_parts.append(f"--- USER DOCUMENT ---\n{text}")
            except:
                pass

        return "\n\n".join(context_parts)

    # ---------------------------
    #           CHAT
    # ---------------------------
    def chat(self, message):
        context = self.retrieve(message)

        if context:
            system_msg = (
                "You are a Senior Indian Legal Advisor. "
                "Answer ONLY using the provided context."
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


# =======================================
#           STREAMLIT UI
# =======================================
st.set_page_config(page_title="Dual RAG Legal Assistant", layout="centered")
st.title("⚖️ Dual RAG Legal Assistant")

# Initialize RAG engine
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = DualLegalRAG()

rag = st.session_state.rag_engine


# -------------------------
#     SIDEBAR UPLOAD
# -------------------------
with st.sidebar:
    st.header("Upload Case PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Process PDF"):
        msg = rag.process_user_upload(uploaded)
        st.info(msg)


# -------------------------
#     CHAT HISTORY
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
#         CHAT BOX
# -------------------------
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.chat(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
