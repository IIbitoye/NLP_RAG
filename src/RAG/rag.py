import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_core.documents import Document
# ----------------------------------------
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# 1. Setup Database Connection
vector_store = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=OpenAIEmbeddings()
)

print("⏳ Booting up Hybrid Retrieval Engine (Vector + Keyword)...")

# A. The Semantic Retriever (Finds concepts & synonyms)
vector_retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 10})

# B. The Keyword Retriever (Finds exact terminology like "manual" or "automated")
db_data = vector_store.get()
all_docs = [Document(page_content=txt, metadata=meta) for txt, meta in zip(db_data['documents'], db_data['metadatas'])]
keyword_retriever = BM25Retriever.from_documents(all_docs)
keyword_retriever.k = 10

# C. The Fusion Engine (Merges and ranks the results)
retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.7, 0.3] # 70% Semantic Focus, 30% Exact Keyword Focus
)
print("✅ Hybrid Retrieval Ready!\n")

# 2. Setup Primary and Fallback LLMs 
# max_retries=0 ensures it fails fast and swaps to Ollama instantly if the API is down
primary_llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2)
fallback_llm = ChatOllama(model="llama3.2", temperature=0)

PROMPT_TEMPLATE = """
You are a Research Assistant. Use ONLY the provided context to answer the question.

CONTEXT:
{context}

QUESTION: 
{question}

INSTRUCTIONS:
1. Extract relevant chunks and quotes from the context.
2. If the context does not contain the answer, say "Insufficient Evidence" and give proper reasons behind it.
3. Cite your sources using the [source_id] provided in the metadata.

OUTPUT FORMAT (JSON):
{{
 "answer": "Your answer...",
  "citations": ["source_01", "source_02", "source_03", "source_04"]
}}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def query_rag(question):
    # A. Retrieve (Now uses the Hybrid Ensemble!)
    docs = retriever.invoke(question)
    
    # B. Format Context
    context_text = "\n\n".join([
        f"[{doc.metadata['source_id']}] {doc.page_content}" 
        for doc in docs
    ])
    
    # C. Generate Answer with Fallback Logic
    try:
        # Try OpenAI first
        chain = prompt | primary_llm
        response = chain.invoke({"context": context_text, "question": question})
    except Exception as e:
        # If OpenAI fails, print a warning and hot-swap to Ollama
        print(f"\n⚠️ OpenAI API Error: {e}")
        print("🔄 Hot-swapping to Local Llama 3.2 Backup Model...\n")
        chain = prompt | fallback_llm
        response = chain.invoke({"context": context_text, "question": question})
    
    return response.content, docs

if __name__ == "__main__":
    q = input("🔎 Enter a research question: ")
    answer, sources = query_rag(q)
    print("\n🤖 AI RESPONSE:\n" + answer)