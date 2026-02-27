# GEMINI.md - Project Context

## Project Overview
This repository is a **RAG (Retrieval-Augmented Generation) Playground** designed for experimenting with various RAG strategies, including chunking, embeddings, retrievals, and evaluations. The project is focused on building an expert assistant for "Insurellm".

### Key Technologies
- **Language:** Python 3.12+
- **LLM Integration:** `litellm` (supports OpenAI, Anthropic, etc.) and `langchain`.
- **Vector Database:** `ChromaDB` (persistent storage).
- **UI Framework:** `Gradio` for both the chat interface and evaluation dashboards.
- **Dependency Management:** `uv` (implied by `uv.lock`) or `pip` via `pyproject.toml`.

### Project Structure
- `interface_v1/`: Main application logic and UIs.
  - `app.py`: Chat assistant interface.
  - `evaluator.py`: Evaluation dashboard for retrieval and answer quality.
  - `data_sets/`: Tools for generating evaluation datasets from knowledge bases.
  - `implementation/`: Standard RAG implementations.
  - `pro_implementation/`: Advanced RAG implementations (e.g., LLM-based chunking).
- `data/knowledge-base/`: Source documents (Markdown) for ingestion.
- `vector_dbs/`: Persistent ChromaDB collections.
- `notebooks/`: Jupyter notebooks for testing chunking, embeddings, and RAG components.
- `test-sets/`: Generated evaluation datasets.

---

## Building and Running

### Setup
1. **Environment Variables:** Create a `.env` file in the root directory with necessary API keys:
   ```env
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   ```
2. **Dependencies:**
   ```bash
   pip install .
   # OR if using uv
   uv sync
   ```

### Running the Application
- **Chat Interface:**
  ```bash
  python interface_v1/app.py
  ```
- **Evaluation Dashboard:**
  ```bash
  python interface_v1/evaluator.py
  ```
- **Ingestion (Vector DB Creation):**
  ```bash
  python interface_v1/pro_implementation/ingest.py
  ```
- **Dataset Generation:**
  ```bash
  python interface_v1/data_sets/main.py --kb data/knowledge-base --output test-sets/my_test.jsonl
  ```

---

## Development Conventions

### Coding Style
- **Modular Logic:** RAG components (ingest, answer, eval) are separated into distinct modules.
- **Pydantic Models:** Used for structured LLM outputs and data validation (see `ingest.py` and `generator.py`).
- **Retry Logic:** `tenacity` is used for handling LLM rate limits and transient errors.

### Data Handling
- **LLM-Based Chunking:** The `pro_implementation` uses GPT models to intelligently chunk documents into headlines, summaries, and original text for better retrieval.
- **Evaluation Metrics:**
  - **Retrieval:** MRR (Mean Reciprocal Rank), nDCG, and Keyword Coverage.
  - **Generation:** Accuracy, Completeness, and Relevance (scored 1-5).

### Workflow
1. **Prototyping:** Use `notebooks/` for initial experiments with new chunking or embedding models.
2. **Implementation:** Update `implementation/` or `pro_implementation/` with refined logic.
3. **Validation:** Generate a dataset using `data_sets/main.py` and run evaluations via `evaluator.py` to measure performance improvements.
