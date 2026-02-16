import sys
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# --- CONFIGURATION ---
print("🧠 Loading Local Models...")

# 1. Embeddings (Must match Ingest)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Vector Store (FIXED: Added collection_name)
vector_store = Chroma(
    collection_name="rag_collection",  # <--- THIS WAS MISSING
    persist_directory=DB_PATH, 
    embedding_function=embeddings
)

# 3. LLM (Ollama)
llm = ChatOllama(model="llama3.2", temperature=0) 

# --- PROMPT ---
prompt = ChatPromptTemplate.from_template("""
You are a helpful research assistant. 
Answer the question based ONLY on the context below.
If you don't know, say "Insufficient Evidence".

Context:
{context}

Question: 
{question}
""")

def chat():
    print("\n🚀 Local RAG System (Ollama Mode) Ready!")
    print(f"📂 Connected to DB at: {DB_PATH}")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("❓ Ask: ")
        if query.lower() in ["exit", "quit"]: break
        
        print("   🔎 Searching local database...")
        
        # Retrieve
        docs = vector_store.similarity_search(query, k=5)
        
        if not docs:
            print("   🔴 No relevant docs found. (Check collection name?)")
            continue
            
        print(f"   ✅ Found {len(docs)} relevant chunks.")
        
        # Context Construction
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # Generate
        print("   🤖 Generating answer with Llama 3.2...")
        chain = prompt | llm
        response = chain.invoke({"question": query, "context": context_text})
        
        print(f"\n🟢 Answer: {response.content}\n")
        print("-" * 50)

if __name__ == "__main__":
    chat()