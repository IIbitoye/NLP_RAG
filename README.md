# Personal Research Portal (PRP) 
**Author:** Iteoluwa Ibitoye
**Course:** AI Systems Management
**Date:** Feb 14, 2026

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

###4. Install Dependencies

Install all required Python packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

##5. Configure Environment Variables

The system requires an OpenAI API key to run.

Create a file named .env in the root directory

Add your API key to the .env file:

OPENAI_API_KEY=sk-proj-your-key-here...


###🚀 How to Run
A. Run the Full Evaluation

To generate the full evaluation report on the 20-query test set, run the main evaluation script. This script uses MMR Reranking and Structured Citations.

```bash
python src/eval/eval.py
```
Output: Prints Q&A to the console and saves the report to outputs/evaluation_results.json.

Logs: Saves detailed retrieval logs (with chunks) to logs/retrieval_logs.json.

B. Interactive Mode (Test Your Own Queries)

To chat with the system and ask your own custom questions about low resource language NLP:

```bash
python src/RAG/query.py
```
Usage: Type your question when prompted. Type exit to quit.

Note: This mode includes an experimental "Query Expansion" feature that im testing for Phase 3 that brainstorms synonyms of the query before searching.

C. Re-Ingest Data (Optional)

If you want to rebuild the database from scratch (e.g., if I added new PDFs to data/):

```bash
python src/ingest/ingest.py
```
Warning: This will delete and recreate the data/chroma_db folder.
