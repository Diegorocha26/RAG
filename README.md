# RAG Playground

A comprehensive Retrieval-Augmented Generation (RAG) sandbox for experimenting with document ingestion, chunking strategies, embedding models, and multi-faceted evaluation metrics.

## 🚀 Overview

The RAG Playground provides a structured environment for the end-to-end development of RAG systems. It features both baseline and advanced implementations, allowing for comparative testing and iterative refinement of RAG pipelines.

## 🛠️ Key Features

- **Interactive Interfaces:**
  - **Chat Interface:** A Gradio-powered application (`interface_v1/app.py`) for interacting with the Insurellm assistant.
  - **Evaluation Dashboard:** A specialized Gradio UI (`interface_v1/evaluator.py`) to visualize and analyze retrieval and generation performance.
- **Advanced RAG (Pro Implementation):**
  - **LLM-Based Ingestion:** Intelligent document processing that chunks text based on semantics, headlines, and summaries for superior retrieval context.
  - **Hybrid Implementations:** Compare standard LangChain-based RAG against custom "Pro" strategies.
- **Evaluation & Benchmarking:**
  - **Retrieval Metrics:** Measure success with MRR (Mean Reciprocal Rank), nDCG, and Keyword Coverage.
  - **Generation Metrics:** Score answer quality on Accuracy, Completeness, and Relevance (1-5 scale).
- **Dataset Generation:** Automated tools (`interface_v1/data_sets/`) to generate high-quality evaluation datasets from diverse source documents.
- **Exploratory Hub:** A collection of Jupyter notebooks dedicated to testing chunking strategies, embedding models, and RAG historical performance.

## 💻 Tech Stack

- **Language:** Python 3.12+
- **Orchestration:** LangChain & LiteLLM (multi-provider support)
- **Vector Database:** ChromaDB
- **UI Framework:** Gradio
- **Dependency Management:** `uv` (preferred) or `pip`

## 📂 Project Structure

- `interface_v1/`: Core application logic, UIs, and evaluation scripts.
  - `implementation/`: Baseline RAG implementations.
  - `pro_implementation/`: Advanced RAG with intelligent ingestion.
  - `data_sets/`: Tools for generating evaluation data.
- `data/knowledge-base/`: Source documents for ingestion.
- `vector_dbs/`: Persistent ChromaDB collections.
- `notebooks/`: Research and development notebooks.
- `test-sets/`: Generated JSONL datasets for evaluation.

## 🚦 Getting Started

### 1. Setup
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### 2. Install Dependencies
```bash
uv sync
# OR
pip install .
```

### 3. Running the Tools
- **Run Chat Assistant:** `python interface_v1/app.py`
- **Run Evaluation Dashboard:** `python interface_v1/evaluator.py`
- **Ingest Documents:** `python interface_v1/pro_implementation/ingest.py`
- **Generate Test Set:** `python interface_v1/data_sets/main.py --kb data/knowledge-base --output test-sets/custom.jsonl`
