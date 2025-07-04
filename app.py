import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import tempfile
import os

try:
    requests.get("https://www.google.com", timeout=5)
    st.success("✅ Internet connection: OK")
except:
    st.error("❌ No internet connection. Please connect to the internet.")

st.set_page_config(page_title="📄 Chat with Your Notes (PDF Q&A Bot)")
st.title("📄 Chat with Your Notes")
st.write("Upload a PDF and ask questions from its content using AI!")

groq_api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")

from groq import Groq

try:
    client = Groq(api_key=groq_api_key)
    client.models.list()  
    st.success("✅ API key is valid.")
except Exception as e:
    st.error(f"❌ API Key Error: {e}")
    st.stop()

uploaded_file = st.file_uploader("📎 Upload a PDF file", type="pdf")

query = st.text_input("💬 Ask a question about your PDF")

if uploaded_file and groq_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    loader = PyPDFLoader(tmp_pdf_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    st.write("Using Groq model: llama3-8b-8192")
    llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("🤖 Generating answer..."):
            result = qa.invoke(query)
            st.success("✅ Answer:")
            st.write(result["result"])

    os.remove(tmp_pdf_path)
