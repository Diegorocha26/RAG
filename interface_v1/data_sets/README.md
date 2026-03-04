# Eval Dataset Generator

A 3-pass LLM pipeline that turns a knowledge base (files or folders) into a curated JSONL evaluation dataset — designed for testing RAG retrieval quality and production LLM systems.

## Output format

Each line in the output JSONL looks like:

```jsonl
{"question": "What was Insurellm's first product?", "keywords": ["Markellm", "first"], "reference_answer": "Insurellm's first product was Markellm, the marketplace connecting consumers with insurance providers.", "category": "direct_fact", "source_files": ["company_overview.md"], "quality_score": 9}
```

| Field | Description |
|---|---|
| `question` | The evaluation question |
| `keywords` | 2–5 words from the answer for automated scoring |
| `reference_answer` | Ground truth answer grounded in the source |
| `category` | Question type (configurable) |
| `source_files` | Filename(s) needed to answer — use for RAG retrieval validation |
| `quality_score` | LLM judge score 1–10 (Pass 3) |

## How the pipeline works

```
Knowledge Base (files/folders)
        │
        ▼
┌─────────────────────────────────────────┐
│  PASS 1 — Per-file extraction           │
│  • Read each file                       │
│  • Extract key facts, entities, dates   │
│  • Generate N candidate questions       │
│  • Output: intermediate/pass1_*.json    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  PASS 2 — Spanning question generation  │
│  • Combine facts from ALL documents     │
│  • Generate cross-document questions    │
│  • Categories: spanning, holistic       │
│  • Output: intermediate/pass2_*.json    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  PASS 3 — LLM-as-judge curation        │
│  • Score every candidate 1–10           │
│  • Filter by quality threshold          │
│  • Select diverse final set             │
│  • Output: output_dataset.jsonl         │
└─────────────────────────────────────────┘
```

## Setup

```bash
uv sync
```
or
```bash
pip install -r requirements.txt
```

Set your API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Basic

```bash
uv run python main.py --kb ./my_knowledge_base
```

### With options

```bash
# Custom total size, output path, and cheaper model
uv run python main.py --kb ./docs --total 30 --model openai/gpt-4o-mini --output my_eval.jsonl

# Use Anthropic Claude
uv run python main.py --kb ./docs --model anthropic/claude-opus-4

# Single file instead of a directory
uv run python main.py --kb ./docs/overview.md

# Skip Pass 3 curation (faster for prototyping)
uv run python main.py --kb ./docs --no-curation

# Verbose logging
uv run python main.py --kb ./docs -v
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--kb` | *(required)* | Path to file or directory |
| `--config` | `config.yaml` | Config YAML path |
| `--output` | from config | Output JSONL path |
| `--total` | from config | Target question count |
| `--model` | from config | LiteLLM model string |
| `--examples` | from config | Examples JSONL path |
| `--temperature` | from config | LLM temperature |
| `--no-curation` | false | Skip Pass 3 |
| `-v` | false | Verbose logging |

## Configuration (`config.yaml`)

### LLM

```yaml
llm:
  model: "openai/gpt-4o"      # Any LiteLLM model string
  temperature: 0.7
  max_tokens: 4096
```

### Dataset size

```yaml
dataset:
  total_questions: 50          # Final dataset size
  min_per_file: 1              # Every file gets at least this many
  max_pct_per_file: 0.20       # No file contributes more than 20%
  spanning_ratio: 0.20         # 20% of total = spanning/holistic questions
```

### Categories (fully configurable)

```yaml
categories:
  direct_fact:
    weight: 0.35               # Proportion of single-doc questions
    multi_doc: false           # false = Pass 1, true = Pass 2
    description: >
      A straightforward question with a single clearly stated answer...

  spanning:
    weight: 0.20
    multi_doc: true            # Will be generated in Pass 2
    description: >
      Requires combining facts from at least two documents...
```

Add, remove, or rename categories freely. The pipeline adapts automatically.

### Curation

```yaml
curation:
  enabled: true
  min_score: 7                 # Questions below this are dropped
  criteria:
    - "Clarity: the question is unambiguous"
    - "Answerability: answer is supported by sources"
    - "Relevance: tests something meaningful"
    - "Diversity: covers different facts"
```

## Supported file formats

| Extension | Library |
|---|---|
| `.md`, `.txt` | stdlib |
| `.pdf` | `pypdf` |
| `.csv` | stdlib |
| `.docx` | `python-docx` |

To add a new format, register a loader function in `loaders.py`:

```python
def load_myformat(path: Path) -> str:
    ...  # return plain text

LOADERS[".myformat"] = load_myformat
```

## Intermediate files

All intermediate results are saved to `intermediate/` for inspection and debugging:

- `intermediate/pass1_extractions.json` — extracted facts per document
- `intermediate/pass1_questions.json` — all single-doc candidates
- `intermediate/pass2_questions.json` — all spanning/holistic candidates

These are preserved even if Pass 3 fails, so you never lose work.

## Using `source_files` for RAG evaluation

The `source_files` field tells you which documents an ideal retriever should have fetched to answer a question. Use it to compute **retrieval recall**:

```python
import json

with open("output_dataset.jsonl") as f:
    questions = [json.loads(line) for line in f]

# For each question: did your retriever return the right docs?
def retrieval_recall(question, retrieved_doc_names):
    required = set(question["source_files"])
    retrieved = set(retrieved_doc_names)
    return len(required & retrieved) / len(required)
```

This lets you decouple RAG retrieval quality from generation quality.
