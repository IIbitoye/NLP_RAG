import sys
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
# CHANGED: Import Local Libraries
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 1. PATH SETUP ---
# Current: src/eval/eval.py -> Root: Phase2_Local/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "data_manifest.csv")

# Outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "evaluation_results_local.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_PATH = os.path.join(LOGS_DIR, "retrieval_logs_local.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- 2. CONFIGURATION (LOCAL) ---
# CHANGED: Use the same embeddings as ingest.py
print("🧠 Loading Local Embeddings (HuggingFace)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# CHANGED: Use Local LLM (Ollama)
print("🦙 Connecting to Ollama (Llama 3.2)...")
llm = ChatOllama(model="llama3.2", temperature=0)

# Connect to DB (FIXED: Added collection_name)
vector_store = Chroma(
    collection_name="rag_collection",
    persist_directory=DB_PATH, 
    embedding_function=embeddings
)

# --- 3. HELPER FUNCTIONS ---
def build_citation_map(manifest_path):
    # (Kept simple for local run - mapping filenames to citations)
    if not os.path.exists(manifest_path):
        return {}
    
    mapping = {}
    try:
        df = pd.read_csv(manifest_path)
        # normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        
        # simple guess for columns
        fname_col = next((c for c in df.columns if 'file' in c), None)
        cite_col = next((c for c in df.columns if 'citation' in c), None)
        
        if fname_col and cite_col:
            for _, row in df.iterrows():
                f = str(row[fname_col]).strip()
                c = str(row[cite_col]).strip()
                mapping[f] = c
    except Exception as e:
        print(f"⚠️ Manifest Error: {e}")
    return mapping

# Load Citation Map
CITATION_MAP = build_citation_map(MANIFEST_PATH)

# Setup Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)

# Prompt (Optimized for Llama 3 JSON)
PROMPT_TEMPLATE = """
You are a Research Assistant. Use ONLY the provided context to answer.

CONTEXT:
{context}

QUESTION: 
{question}

INSTRUCTIONS:
1. Answer clearly based ONLY on the context.
2. If the context is empty or irrelevant, strictly return "Insufficient Evidence".
3. Return the output as a raw JSON object (no markdown formatting).

JSON FORMAT:
{{
  "answer": "Your answer string here...",
  "source_files": ["filename1.pdf", "filename2.pdf"]
}}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def run_query(question):
    print(f"\n🔵 Query: {question}")
    start_time = time.time()
    
    # 1. Retrieve
    docs = retriever.invoke(question)
    
    # 2. Prepare Context
    context_text = ""
    retrieved_log = []
    found_files = set()
    
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
        content = doc.page_content.replace("\n", " ")
        context_text += f"SOURCE: {source}\nCONTENT: {content}\n\n"
        
        found_files.add(source)
        retrieved_log.append({
            "source": source,
            "text": content[:200] + "..."
        })
    
    # 3. Generate
    chain = prompt | llm
    
    answer_text = "Error"
    citations = []
    
    try:
        response = chain.invoke({"context": context_text, "question": question})
        # Clean cleanup for Llama 3
        clean_content = response.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
            
        data = json.loads(clean_content)
        answer_text = data.get("answer", "No answer found")
        
        # Map filenames to Citations
        raw_files = data.get("source_files", [])
        citations = [CITATION_MAP.get(f, f) for f in raw_files]
        
    except json.JSONDecodeError:
        # Fallback if Llama returns plain text
        answer_text = response.content
        citations = [CITATION_MAP.get(f, f) for f in list(found_files)]
    except Exception as e:
        answer_text = f"Error: {e}"

    elapsed = time.time() - start_time
    
    print(f"🟢 Answer: {answer_text[:100]}...")
    print(f"📚 Sources: {citations}")
    
    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "retrieved_chunks": retrieved_log,
        "time_taken": round(elapsed, 2)
    }

# --- 4. QUESTIONS ---
questions = [
    "What is Masakhane?",
    "What are the main findings of the AfroBench paper?",
    "How does 'NLLB' handle low-resource languages?",
    "What is the 'Bitter Lesson' described in the context?"
    # Direct
    "What specific failures does the 'AfroBench' paper identify in current LLMs?",
    "According to Conneau (2020), how does XLM-R compare to mBERT?",
    "What are the three main challenges in preserving cultural identity according to Anik (2025)?",
    "How does the 'Cheetah' paper propose to handle 517 African languages?",
    "What metrics were used to evaluate the 'NaijaSenti' corpus?",
    "Does the 'Localising SA official languages' paper recommend manual or automated collection?",
    "What is the 'Bitter Lesson' described by Wu et al. (2025)?",
    "List the datasets used in the 'IrokoBench' benchmark.",
    "What is the main contribution of the 'No Language Left Behind' project?",
    "How does 'AfriCOMET' improve upon standard COMET metrics?",
    
    # Synthesis
    "Compare the approaches of 'Masakhane' and 'NLLB' regarding community involvement.",
    "What common biases do 'CultureVLM' and 'Global MMLU' identify in multilingual models?",
    "Synthesize the findings on 'Code-Switching' from Terblanche (2024) and any other relevant paper.",
    "Do 'AfroBench' and 'IrokoBench' agree on the performance of GPT-4 for African languages?",
    "How do 'NileChat' and 'Jawaher' differ in their approach to Arabic dialects?",

    # Edge Cases
    "What does the corpus say about 'Quantum Computing in Yoruba'?", 
    "Does the 'WAXAL' paper discuss speech synthesis for Martian languages?",
    "Find evidence for the claim that 'LLMs are perfect translators'.",
    "What is the specific learning rate used in the 'DeepSeek-V3' paper?",
    "Does the corpus contain the personal email address of the author 'Adebara'?"

]

if __name__ == "__main__":
    print("🚀 Starting Local Evaluation...")
    
    full_results = []    # Stores detailed data (with chunks) for logs
    summary_results = [] # Stores clean data (just Q&A) for the report
    
    for q in questions:
        # Run the query
        res = run_query(q)
        
        # Add to full logs
        full_results.append(res)
        
        # Add to summary (exclude the heavy 'retrieved_chunks' to keep it clean)
        summary_results.append({
            "question": res["question"],
            "answer": res["answer"],
            "citations": res["citations"],
            "time_taken": res["time_taken"]
        })
        
    # 1. Save Clean Report (for grading)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary_results, f, indent=2)
        
    # 2. Save Detailed Logs (for debugging/proof) <--- THIS WAS MISSING
    with open(LOGS_PATH, "w") as f:
        json.dump(full_results, f, indent=2)
        
    print(f"\n✅ Evaluation Complete.")
    print(f"📄 Clean Report: {SUMMARY_PATH}")
    print(f"🪵  Detailed Logs: {LOGS_PATH}")