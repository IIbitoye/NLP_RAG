# Personal Research Portal (PRP) 
**Author:** Iteoluwa Ibitoye

**Course:** AI Systems Management


# 🌍 Low-Resource NLP RAG: A Research Assistant for African Languages

This is a retrieval-augmented generation (RAG) pipeline designed to answer research questions about **Low-Resource NLP and African Languages**. It ingests 30+ academic papers, chunks them, and uses OpenAI's GPT-4o with a "Retrieval-First" prompt strategy to ensure grounded answers.

## 🚀 Key Features
* **Domain-Specific Corpus:** Indexed 34 high-impact papers (Masakhane, NLLB, AfroBench).
* **MMR Reranking:** Uses Maximal Marginal Relevance to retrieve diverse perspectives for synthesis questions.
* **Trusted Citations:** Automatically maps vector chunks to formal academic citations (e.g., `(Adebara et al., 2022)`).
* **Dual Logging:** Generates clean reports for users and detailed retrieval logs for debugging.

## 🛠️ Architecture
* **Ingestion:** `PyPDFLoader` + `RecursiveCharacterTextSplitter` (Chunk size: 1000, Overlap: 200).
* **Embedding:** OpenAI `text-embedding-3-small`.
* **Vector Store:** ChromaDB (Persistent).
* **Retrieval:** MMR (`k=12`, `fetch_k=20`) to reduce redundancy.
* **Generation:** GPT-4o with strict "insufficient evidence" guardrails.

📊 Evaluation Results
The system was evaluated on a diverse set of 20 queries (Direct Fact Retrieval, Multi-Paper Synthesis, and Hallucination Tests).

Metric	Score	Notes
Success Rate	90% (18/20)	High recall on both specific metrics and abstract comparisons.
Safety Score	100% (5/5)	Correctly refused to answer out-of-scope queries (e.g., "Martian Languages").
Latency	~2.7s	Average end-to-end processing time.


## 📂 Project Structure
```text
Phase 2_iibitoye/
├── data/ 
│   ├── data_manifest.csv       # Metadata (Filename -> Citation mapping)
|   |--raw/pdfs
|   |--chroma_db/
├── logs/
│   ├── retrieval_logs.json     # Detailed logs including retrieved chunks for auditing
├── src/
│   ├── ingest/
│   │   ├── ingest.py           # Parses PDFs and builds the ChromaDB vector store
│   ├── eval/
│   │   ├── eval.py             # Main Evaluation Script (Implements MMR + Logging)
│   │   ├── json_to_csv.py      # Utility to convert JSON reports to CSV for grading
│   ├── RAG/
│   │   ├── rag.py              # Core RAG logic module
│   │   ├── query.py            # Interactive CLI for testing custom queries
├── outputs/
│   ├── evaluation_results.json # Clean Q&A Report (JSON format)
│   ├── evaluation_results.csv  # Clean Q&A Report (CSV format)
├── requirements.txt            # Python dependencies
└── .env                        # (Not uploaded) Contains OPENAI_API_KEY
└──Phase2_Local                <-- Seperate but exactly identical folder using Ollama LLM's instead of OpenAi so there's  no need for API keys
|   ├── .env                  
|   ├── requirements.txt      <-- (Local Dependencies: ollama, huggingface)
|   │
|   ├── data/                 <-- (Local Data Copy)
    │   ├── raw/              <-- (Copy of PDFs)
    │   ├── data_manifest.csv <-- (Copy of Manifest)
    │   └── chroma_db/        <
    │
    ├── src/                  <-- (Local Code - Ollama)
    │   ├── ingest/ingest.py  <-- (Uses HuggingFace Embeddings)
    │   ├── RAG/rag.py        <-- (Uses Llama 3.2)
    │   ├── RAG/query.py
    │   └── eval/eval.py
    │
    ├── outputs/              <-- (Local Results)
    └── logs/                 <-- (Local Logs)
```

## 🛠️ Setup & Installation

Follow these steps to set up the project locally.

### 1. Prerequisites
* **Python 3.10+** (Recommended)
* **Git**
* **OpenAI API Key** (Required for Embeddings and LLM)

### 2. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Phase2_iibitoye.git](https://github.com/YOUR_USERNAME/Phase2_iibitoye.git)
cd Phase2_iibitoye
```

##3. Create a Virtual Environment (Optional but Recommended)
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

# Windows
```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Install Dependencies

Install all required Python packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

## 5. Configure Environment Variables

# The system requires an OpenAI API key to run! 
# (Note: AI Mod dev TA's can run the research portal completely using my submitted zipped folder)

Create a file named .env in the root directory

Add your API key to the .env file:

OPENAI_API_KEY=sk-proj-your-key-here...


### 🚀 How to Run
# A. Run the Full Evaluation

To generate the full evaluation report on the 20-query test set, run the main evaluation script. This script uses MMR Reranking and Structured Citations.

```bash
python src/eval/eval.py
```
Output: Prints Q&A to the console and saves the report to outputs/evaluation_results.json.

Logs: Saves detailed retrieval logs (with chunks) to logs/retrieval_logs.json.

# B. Interactive Mode (Test Your Own Queries)

To chat with the system and ask your own custom questions about low resource language NLP:

```bash
python src/RAG/query.py
```
Usage: Type your question when prompted. Type exit to quit.

Note: This mode includes an experimental "Query Expansion" feature that im testing for Phase 3 that brainstorms synonyms of the query before searching.

# C. Re-Ingest Data (Optional)

If you want to rebuild the database from scratch (e.g., if I added new PDFs to data/):

```bash
python src/ingest/ingest.py
```
Warning: This will delete and recreate the data/chroma_db folder.

# 📂 Alternative Version: Local Execution (No API Keys)

For graders or users who wish to run this system **locally** without OpenAI API keys, a fully local implementation is provided in the `Phase2_Local/` folder.

**Path:** `./Phase2_Local/`


### **1. Architecture Note**
This local version functions **identically** to the main submission (RAG Pipeline: Ingest $\rightarrow$ Retrieve $\rightarrow$ Generate). The only difference is the model components:
* **Embeddings:** Uses `HuggingFace (all-MiniLM-L6-v2)` instead of `OpenAI`.
* **LLM:** Uses `Ollama (Llama 3.2)` instead of `GPT-4o`.

### **2. Setup Instructions**
Since this version uses local models, it requires a different set of dependencies.

**A. Prerequisites (One-Time Setup)**
1.  **Install Ollama:** Download from [ollama.com](https://ollama.com).
2.  **Pull the Model:** Open your terminal and run:
    ```bash
    ollama pull llama3.2
    ```
    *(This downloads the 2GB model weight file required for the LLM to run).*

**B. Installation**
Navigate to the local folder and install the specific local dependencies:
```bash
cd Phase2_Local
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # <--- Installs langchain-ollama, huggingface, etc.
```
3.  Ensure Ollama is running (`ollama serve`).
4.  Run `python src/RAG/query.py`. for your personal queries.


Everything else runs the same but the major differences are below
**Key Differences:**
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) instead of OpenAI.
* **LLM:** Ollama (`Llama 3.2`) instead of GPT-4o.
* **Performance:** Faster and free, but with lower reasoning accuracy due to model size (3B vs 1T parameters).

