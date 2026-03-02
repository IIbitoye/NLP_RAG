import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CHANGED: Pointing to "data/raw" instead of "data/pdfs"
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

def create_vector_db():
    print(f"🚀 Starting Local Ingestion...")
    print(f"📂 Looking for PDFs in: {DATA_PATH}")

    # 1. Load Data
    # Check if directory exists first
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: The directory '{DATA_PATH}' does not exist.")
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print(f"❌ ERROR: No PDFs found in '{DATA_PATH}'!")
        return
        
    print(f"   -> Loaded {len(documents)} pages.")

    # 2. Split Data (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Split into {len(chunks)} chunks.")

    # 3. Initialize Local Embeddings
    print("🧠 Loading HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Reset DB
    if os.path.exists(DB_PATH):
        print("🗑️  Deleting old database...")
        shutil.rmtree(DB_PATH)

    # 5. Create Vector Store (Batch Processing)
    print(f"💾 Saving to ChromaDB at {DB_PATH}...")
    
    # Initialize Chroma
    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    
    # Add in batches to prevent memory issues
    batch_size = 100
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"   Processing batch {i}/{total_chunks}...")
        vector_store.add_documents(documents=batch)

    print("✅ Local Database Built Successfully!")

if __name__ == "__main__":
    create_vector_db()