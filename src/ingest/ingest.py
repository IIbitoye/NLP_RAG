import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "data_manifest.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

def ingest_data():
    print("Loading Data Manifest...")
    try:
        manifest = pd.read_csv(MANIFEST_PATH)
    except FileNotFoundError:
        print("❌ Error: data_manifest.csv not found.")
        return

    documents = []

    print(f"Found {len(manifest)} papers. Starting ingestion...")
    
    for index, row in manifest.iterrows():
        file_path = os.path.join(DATA_PATH, row['filename'])
        
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File {row['filename']} not found. Skipping.")
            continue

        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Attach Metadata from Manifest to every single page
            for doc in docs:
                doc.metadata['source_id'] = row['source_id']
                doc.metadata['citation'] = row['citation'] # e.g. "(Smith, 2023)"
                doc.metadata['title'] = row['title']
            
            documents.extend(docs)
            print(f"   ✅ Loaded {row['filename']}")
            
        except Exception as e:
            print(f"   ❌ Failed to load {row['filename']}: {e}")

    # 3. Chunking (Splitting text into pieces)
    # We use a 1000 character chunk with 200 overlap to keep context
    print("✂️  Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} text chunks.")

    # 4. Embed & Save to Vector DB
    print("💾 Saving to Vector Database (this may take a minute)...")
    
    # We use ChromaDB (local file) and OpenAI Embeddings
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=DB_PATH
    )
    
    print(f"🚀 Success! Database created at {DB_PATH}")

if __name__ == "__main__":
    ingest_data()