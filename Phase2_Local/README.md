# Phase 2 Technical Appendix: Local RAG Implementation
Deployment Mode: Localized Edge Inference (Ollama-based)

This directory contains a standalone, localized version of the Research Portal. While the primary submission utilizes GPT-4o for maximum reasoning density, this implementation demonstrates the system's portability and its ability to function within a Privacy-Preserving and Zero-Egress environment.

## 🏗️ Technical Specifications
To transition from cloud-based APIs to local execution, the architectural components were swapped for high-efficiency, open-source equivalents:

Large Language Model: Llama-3.2-3B (via Ollama)

Rationale: Provides an optimal balance of latency and linguistic performance for local CPU/GPU inference.

Vector Embeddings: sentence-transformers/all-MiniLM-L6-v2 (via HuggingFace)

Rationale: A 384-dimensional dense vector model optimized for semantic similarity without requiring massive VRAM overhead.

Orchestration: LangChain (v0.3) utilizing langchain-ollama and langchain-huggingface integration libraries.

## 🛠️ Local Environment Setup
1. Model Provisioning

This system requires the Ollama runtime to manage local model weights.


# Ensure Ollama is installed (ollama.ai)
```Bash
ollama pull llama3.2
```

# 2. Dependency Management
The local implementation utilizes a distinct dependency tree to avoid library conflicts with the OpenAI-based primary system.

```
cd Phase2_Local
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# 3. Pipeline Execution
Knowledge Ingestion:
python src/ingest/ingest.py
(Parses PDFs and initializes the ChromaDB instance using local embeddings.)

# Inference/Retrieval:
```
python src/RAG/query.py
```
(Launches the interactive CLI for RAG-augmented querying.)

## 📊 Design Considerations & Trade-offs
Feature	Primary (GPT-4o)	Local (Llama 3.2)
Data Privacy	Cloud-processed	100% Local / On-prem
Inference Cost	Token-based (Variable)	Zero ($0.00)
Reasoning Depth	High (Multi-step synthesis)	Moderate (Fact retrieval)
Connectivity	Requires Internet	Offline-capable

# A Note on Evaluation Performance
As noted in the main report, the evaluation metrics for this local instance will show a slight variance in synthesis quality compared to the GPT-4o results. This is an expected result of reducing the model parameter count from >1T to 3B. However, the retrieval accuracy (the ability to find the correct document chunk) remains consistent across both implementations.

## 📂Local Directory Structure
Plaintext
Phase2_Local/
├── requirements.txt      # Specific local dependencies (HuggingFace/Ollama)
├── data/
│   ├── raw/              # PDF Corpus
│   └── chroma_db/        # Local Vector Store (Generated)
├── src/
│   ├── ingest/           # Local embedding logic
│   └── RAG/              # Local inference & prompt logic
└── outputs/              # Local evaluation results
