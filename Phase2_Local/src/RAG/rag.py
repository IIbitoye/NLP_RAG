import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

class LocalRAGSystem:
    def __init__(self):
        print("⏳ Initializing Local RAG System...")
        
        # 1. Embeddings
        print("   -> [1/3] Loading Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. ChromaDB (FIXED: Added collection_name)
        print(f"   -> [2/3] Connecting to Database at {DB_PATH}...")
        self.vector_store = Chroma(
            collection_name="rag_collection",  # <--- THIS WAS MISSING
            persist_directory=DB_PATH,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("      ✅ Connected.")
        
        # 3. Ollama
        print("   -> [3/3] Connecting to Ollama (Llama 3.2)...")
        self.llm = ChatOllama(model="llama3.2", temperature=0)

        # 4. Prompt
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant.
        Answer the question based ONLY on the context provided below.
        If the answer is not in the context, say "Insufficient Evidence".

        Context:
        {context}

        Question: 
        {question}
        """)
        
        # 5. Chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question):
        print(f"\n🔎 Searching for: '{question}'...")
        # Debug: Print first 50 chars of retrieved docs to prove it found something
        docs = self.retriever.invoke(question)
        if not docs:
            print("   ⚠️ WARNING: No documents found! DB might be empty.")
        else:
            print(f"   ✅ Found {len(docs)} relevant chunks.")
            
        return self.chain.invoke(question)

if __name__ == "__main__":
    rag = LocalRAGSystem()
    
    # Test Query
    q = "What is Masakhane?"
    print(f"❓ Asking: {q}")
    
    response = rag.query(q)
    print(f"\n🟢 ANSWER:\n{response}")