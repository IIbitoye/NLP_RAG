# 🌍 Personal Research Portal (PRP)
**Author:** Iteoluwa Ibitoye  
**Course:** AI Systems Management  

## 📖 Overview
This is a retrieval-augmented generation (RAG) pipeline and interactive web portal designed to answer research questions about **Low-Resource NLP and African Languages**. It ingests 30+ academic papers, chunks them, and uses OpenAI's GPT-4o with a "Retrieval-First" prompt strategy to ensure highly grounded, citation-backed answers.

## 🚀 Key Features

Domain-Specific Corpus: Indexed 34 high-impact papers (Masakhane, NLLB, AfroBench).
MMR Reranking: Uses Maximal Marginal Relevance to retrieve diverse perspectives for synthesis questions.
Trusted Citations: Automatically maps vector chunks to formal academic citations (e.g., (Adebara et al., 2022)).
Dual Logging: Generates clean reports for users and detailed retrieval logs for debugging.

### ✨ What's New in Phase 3
* **Interactive Streamlit UI:** A conversational interface with real-time citation tracking and session memory.
* **Automated Artifact Generation:** Instantly exports Evidence Tables (CSV), Annotated Bibliographies (APA Markdown), and 800+ word Synthesis Memos (Markdown).
* **LLM Hot-Swapping:** A defensive engineering fallback that automatically routes requests to a local `Llama 3.2` model if the OpenAI API fails or hits rate limits.
* **Evaluation Dashboard:** In-app metrics tracking system latency, groundedness, and citation correctness.

---

## 🛠️ Architecture
* **Ingestion:** `PyPDFLoader` + `RecursiveCharacterTextSplitter` (Chunk size: 1000, Overlap: 200).
* **Embedding:** OpenAI `text-embedding-3-small`.
* **Vector Store:** ChromaDB (Persistent local database).
* **Retrieval:** MMR (`k=12`, `fetch_k=20`) to reduce redundancy and enforce diverse context.
* **Generation:** `GPT-4o` (Primary) with `Llama 3.2` (Fallback), featuring strict "insufficient evidence" guardrails.

---

## 🔬 Methodology & Evaluation

### Corpus Selection Process
The corpus consists of 34 high-impact academic papers focusing specifically on low-resource Natural Language Processing for African languages. Papers were selected based on their relevance to multilingual benchmarking (e.g., AfroBench), community-driven NLP initiatives (e.g., Masakhane), and machine translation architectures (e.g., NLLB). The selection aims to provide a comprehensive overview of the current challenges, methodologies, and datasets unique to this domain.

### Groundedness Metric Definition
Groundedness measures the extent to which the model's generated answer is directly supported by the retrieved context chunks. It is scored manually on a 1-4 scale:
* **1:** Complete hallucination or contradiction.
* **2:** Mentions topic but relies heavily on external parametric knowledge.
* **3:** Mostly supported, but contains minor logical leaps.
* **4:** Fully supported, factually accurate, and entirely traceable to the provided chunks.

### 📊 Evaluation Results
The system was evaluated on a diverse set of 20 queries (Direct Fact Retrieval, Multi-Paper Synthesis, and Hallucination Tests).

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Success Rate** | 90% (18/20) | High recall on both specific metrics and abstract comparisons. |
| **Safety Score** | 100% (5/5) | Correctly refused to answer out-of-scope edge cases. |
| **Avg. Groundedness** | 3.90 / 4.0 | Exceptional adherence to retrieved context. |
| **Latency** | ~3.18s | Average end-to-end processing time per query. |

### Explicit Failure Case Analysis
During evaluation, the system exhibited two primary failure modes:
1. **Cross-Document Synthesis Gap:** For queries requiring comparisons between multiple distinct models, MMR retrieval sometimes failed to pull adequate chunks for *both* papers simultaneously, triggering an 'Insufficient Evidence' guardrail.
2. **Vocabulary Mismatch:** Queries using terminology that heavily deviated from the authors' specific phrasing (e.g., searching "manual collection" instead of "crowdsourcing") occasionally resulted in missed retrievals due to the rigid semantic matching of the embedding model.

---

## 📂 Project Structure
```text
Phase 2_iibitoye/
├── app.py                      # MAIN STREAMLIT APPLICATION (Phase 3 UI)
├── data/ 
│   ├── data_manifest.csv       # Metadata (Filename -> Citation mapping)
│   ├── raw/pdfs/               # Original 34 academic papers
│   └── chroma_db/              # Persistent vector database
├── outputs/
│   ├── evaluation_grading_sheet2.csv # Full grading metrics
│   └── chat_history.json       # Persistent session memory for UI
├── src/
│   ├── ingest/ingest.py        # Parses PDFs and builds ChromaDB
│   ├── eval/eval.py            # Main Evaluation Script (MMR + Logging)
│   └── RAG/
│       ├── rag.py              # Core RAG logic with LLM Fallback
│       └── query.py            # CLI query tool with Multi-Query Expansion
├── requirements.txt            # Python dependencies
└── .env                        # Contains OPENAI_API_KEY

## 🛠️ Setup & Installation

Follow these steps to set up the project locally.

### 1. Prerequisites
* **Python 3.10+** (Recommended)
* **Git**
* **OpenAI API Key** (Required for Embeddings and LLM)

### 2. Clone the Repository
```bash
git clone [https://github.com/iibitoye/Phase2_iibitoye.git](https://github.com/YOUR_USERNAME/Phase2_iibitoye.git)
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
# A. Launch the Web Portal (Main Feature)
To launch the full Phase 3 interactive UI for direct querying, and click on "esport artifacts" to generatte synthesis memo, annotated bibliography, and evidence table artifacts, and view the evaluation dashboard:

``` bash
streamlit run app.py
```
# B. Run the Full Evaluation

To generate the full evaluation report on the 20-query test set, run the main evaluation script. This script uses MMR Reranking and Structured Citations.

```bash
python src/eval/eval.py
```
Output: Prints Q&A to the console and saves the report to outputs/evaluation_results.json.

Logs: Saves detailed retrieval logs (with chunks) to logs/retrieval_logs.json.

# C. Interactive Mode (Test Your Own Queries)

To chat with the system and ask your own custom questions about low resource language NLP:

```bash
python src/RAG/query.py
```
Usage: Type your question when prompted. Type exit to quit.

Note: This mode includes an experimental "Query Expansion" feature that brainstorms and uses synonyms of the query before searching.

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

