import sys
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "data_manifest.csv")


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "evaluation_results_final3.json")


LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_PATH = os.path.join(LOGS_DIR, "retrieval_logs3.json")


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- 1. SMART CITATION MAPPING ---
def build_citation_map(vector_store, manifest_path):
    print("🗺️  Building citation map...")
    filename_to_citation = {}
    try:
        df = pd.read_csv(manifest_path)
        # Normalize column names
        df.columns = [c.strip() for c in df.columns] 
        
        fname_col, cite_col = None, None
        for col in df.columns:
            if col.lower() in ['filename', 'file name', 'file', 'name']: fname_col = col
            if col.lower() in ['citation', 'citations', 'cite']: cite_col = col
        
        if fname_col and cite_col:
            for _, row in df.iterrows():
                fname = os.path.basename(str(row[fname_col]).strip())
                filename_to_citation[fname] = str(row[cite_col])
    except Exception as e:
        print(f"⚠️ Warning: Manifest error: {e}")

    try:
        data = vector_store.get()
        id_to_citation = {}
        for meta in data['metadatas']:
            if not meta: continue
            s_id = meta.get('source_id')
            fname = meta.get('filename') or meta.get('source')
            if s_id and fname:
                fname = os.path.basename(fname).strip()
                id_to_citation[s_id] = filename_to_citation.get(fname, fname)
        print(f"✅ Mapped {len(id_to_citation)} citations.")
        return id_to_citation
    except Exception:
        return {}


vector_store = Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings())
SOURCE_ID_TO_CITATION = build_citation_map(vector_store, MANIFEST_PATH)

# --- 2. RAG SETUP ---
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 12, "fetch_k": 20, "lambda_mult": 0.7}
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

PROMPT_TEMPLATE = """
You are a Research Assistant. Use ONLY the provided context to answer the question.

CONTEXT:
{context}

QUESTION: 
{question}

INSTRUCTIONS:
1. Answer clearly and concisely.
2. If the context has no answer, say "Insufficient Evidence".
3. Cite your sources using the [source_id] found in the context (e.g., [source_05]).
4. If the context has multiple sources, cite the relevant ones.

OUTPUT FORMAT (JSON):
{{
  "answer": "Your answer...",
  "citations": ["source_01", "source_05"]
}}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def run_query(question):
    print(f"\n🔵 Query: {question}")
    start_time = time.time()
    
    # 1. Retrieve
    docs = retriever.invoke(question)
    
    # 2. Process Context & Log Chunks
    context_text = ""
    retrieved_chunks_log = [] 
    
    for doc in docs:
        s_id = doc.metadata.get('source_id', 'Unknown')
        content = doc.page_content
        context_text += f"[{s_id}] {content}\n\n"
        
        retrieved_chunks_log.append({
            "source_id": s_id,
            "citation": SOURCE_ID_TO_CITATION.get(s_id, "Unknown"),
            "text_snippet": content  
        })
    
    # 3. Generate
    chain = prompt | llm
    try:
        response = chain.invoke({"context": context_text, "question": question})
        content = response.content.replace("```json", "").replace("```", "")
        result_json = json.loads(content)
        
        answer = result_json.get("answer", "Error parsing answer")
        raw_ids = result_json.get("citations", [])
        for s_id, readable_cite in SOURCE_ID_TO_CITATION.items():
            answer = answer.replace(f"[{s_id}]", f"({readable_cite})")


        # Convert IDs to Real Citations
        readable_citations = []
        for rid in raw_ids:
            clean_id = rid.replace("[", "").replace("]", "").strip()
            citation = SOURCE_ID_TO_CITATION.get(clean_id, clean_id)
            if citation not in readable_citations:
                readable_citations.append(citation)

    except Exception as e:
        answer = f"Error: {str(e)}"
        readable_citations = []
        raw_ids = []

    elapsed = time.time() - start_time
    
    # Print to Terminal
    print(f"🟢 Answer: {answer}")
    if readable_citations:
        print(f"📚 Sources: {', '.join(readable_citations)}")
    else:
        print("📚 Sources: None")
    print("-" * 60)


    return {
        "question": question,
        "answer": answer,
        "citations_readable": readable_citations,
        "citations_raw": raw_ids,
        "retrieved_chunks": retrieved_chunks_log, # <--- The requirement
        "time_taken": round(elapsed, 2)
    }

# --- 3. THE QUESTIONS --
questions = [
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
    print("🚀 Starting Final Evaluation Run (MMR + Logging)...")
    full_results = []
    
    for q in questions:
        full_results.append(run_query(q))
    
    
    # 1. Summary (Clean for Report)
    summary_results = []
    for res in full_results:
        summary_results.append({
            "question": res["question"],
            "answer": res["answer"],
            "citations": res["citations_readable"],
            "time_taken": res["time_taken"]
        })
    
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary_results, f, indent=2)
        
    # 2. Detailed Logs
    with open(LOGS_PATH, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n✅ Done! Files Saved:")
    print(f"📄 Report Data: {SUMMARY_PATH}")
    print(f"🪵  Run Logs:   {LOGS_PATH}")