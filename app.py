import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

import os
import time
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Research Paper Q-A Assistant", page_icon="🤖", layout="wide")
st.sidebar.header("⚙️ Settings")

# Select specific PDF
pdf_folder = "./research_papers"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
selected_pdf = st.sidebar.selectbox("📑 Select Research Paper", pdf_files)

# Model parameters
temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.3, step=0.1)
max_tokens = st.sidebar.slider("🧩 Max Output Tokens", 256, 4096, 1024, step=128)

# Model selection
model_choice = st.sidebar.selectbox("🧠 Choose LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])

# Initialize GROQ LLM
llm = ChatGroq(model=model_choice, groq_api_key=groq_api_key, temperature=temperature, max_tokens=max_tokens)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based **only** on the provided context.
    Be clear, concise, and accurate.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Create vector embeddings
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

        # Load only the selected PDF
        pdf_path = os.path.join(pdf_folder, selected_pdf)
        st.session_state.loader = PyPDFLoader(pdf_path)
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.warning("⚠️ No PDF files found in the specified folder!")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:5])
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, embedding=st.session_state.embeddings
        )
        st.success(f"Vector Database created for **{selected_pdf}**✅")


# Streamlit UI
st.title("🤖 Research Paper Q-A Assistant")
user_prompt = st.text_input("💬 Enter your question below:", placeholder="Start typing...")

if st.button("📄 Create Document Embeddings"):
    create_vector_embeddings()

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please create document embeddings first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed = time.process_time() - start

        st.subheader("🧠 Answer:")
        st.write(response["answer"])
        st.caption(f"⏱️ Response Time: {elapsed:.2f} sec")

        with st.expander("📚 Retrieved Context Documents"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content[:800] + "...")
