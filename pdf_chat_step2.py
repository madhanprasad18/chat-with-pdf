import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  


groq_api_key = input("🔑 Paste your Groq API key here: ")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

os.environ["GROQ_API_KEY"] = groq_api_key

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama3-8b-8192"

)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

print("\n🤖 Ask me anything from your PDF (type 'exit' to quit):\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa(query)
    print("\n📘 Answer:", result["result"])
