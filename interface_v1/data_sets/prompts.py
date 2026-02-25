"""
prompts.py
----------
All LLM prompts used across the three passes, centralized for easy editing.
"""

import json


# TODO: looks good but check if pydantic would be better for structured outputs instead of leaving it to the LLM
# TODO: for pass1, get the examples based on the categories istead of assuming its the first three (like what is done in pass 2).
#       this also goes for the "categories" part of the output. generate it based on the config file, don't assume it 
# TODO: for pass2 it's good that it checks for spanning and holistic examples, but let this be determined by multi doc: true, since the name of the categories can change 

# ── Shared system prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert dataset curator helping build evaluation sets 
for testing LLM systems and RAG (Retrieval-Augmented Generation) pipelines.
Your outputs must be precise, grounded strictly in the provided documents, 
and formatted as valid JSON with no extra prose."""


# ── Pass 1: Extraction ──────────────────────────────────────────────────────────

def pass1_extraction_prompt(
    filename: str,
    content: str,
    single_doc_categories: dict,
    examples: list[dict],
    n_questions: int,
) -> str:
    """
    Prompt for Pass 1: extract facts and generate raw candidate questions
    for a single document.
    """
    categories_block = "\n".join(
        f'  - "{name}": {meta["description"].strip()}'
        for name, meta in single_doc_categories.items()
    )

    examples_block = json.dumps(examples[:3], indent=2) if examples else "[]"

    return f"""You are analyzing the following document from a knowledge base.

    DOCUMENT FILENAME: {filename}
    DOCUMENT CONTENT:
    ---
    {content}
    ---

    YOUR TASK:
    1. Extract the most important facts, entities, dates, numbers, and relationships from this document.
      Focus on information that would be useful for retrieval evaluation.

    2. Generate exactly {n_questions} candidate evaluation questions based ONLY on facts in this document.
      Spread questions across these categories:
      {categories_block}

    EXAMPLE OUTPUT STYLE (from a different knowledge base — do not copy these):
    {examples_block}

    OUTPUT FORMAT — return a single JSON object with this exact structure:
    {{
      "filename": "{filename}",
      "extracted_facts": [
        "fact 1 as a short sentence",
        "fact 2 as a short sentence"
      ],
      "candidate_questions": [
        {{
          "question": "...",
          "keywords": ["keyword1", "keyword2"],
          "reference_answer": "...",
          "category": "direct_fact | temporal | comparative | ...",
          "source_files": ["{filename}"]
        }}
      ]
    }}

    RULES:
    - Every reference_answer must be fully supported by the document content above.
    - Do NOT invent facts. If the document doesn't contain enough info for a category, skip that category.
    - keywords should be 2-5 specific words or numbers from the answer that an evaluator can use to verify correctness.
    - Be precise: prefer specific numbers, names, and dates over vague statements.
    """


# ── Pass 2: Spanning / Holistic generation ──────────────────────────────────────

def pass2_spanning_prompt(
    all_facts: list[dict],
    multi_doc_categories: dict,
    examples: list[dict],
    n_spanning: int,
) -> str:
    """
    Prompt for Pass 2: generate spanning and holistic questions using
    the extracted facts from all documents.
    """
    # Build a compact facts summary per document
    # TODO: check this parsing
    facts_block = ""
    for doc in all_facts:
        facts_block += f"\n### {doc['filename']}\n"
        for fact in doc["extracted_facts"]:
            facts_block += f"  - {fact}\n"

    categories_block = "\n".join(
        f'  - "{name}": {meta["description"].strip()}'
        for name, meta in multi_doc_categories.items()
    )

    spanning_examples = [e for e in examples if e.get("category") in ("spanning", "holistic")]
    examples_block = json.dumps(spanning_examples[:3], indent=2) if spanning_examples else "[]"

    return f"""You are creating multi-document evaluation questions for a RAG system.

    Below are extracted facts from multiple documents in the knowledge base:
    {facts_block}

    YOUR TASK:
    Generate exactly {n_spanning} questions that REQUIRE information from MORE THAN ONE document.
    Focus on these categories:
    {categories_block}

    EXAMPLE OUTPUT STYLE (from a different knowledge base — do not copy these):
    {examples_block}

    OUTPUT FORMAT — return a JSON array:
    [
      {{
        "question": "...",
        "keywords": ["keyword1", "keyword2"],
        "reference_answer": "...",
        "category": "spanning | holistic",
        "source_files": ["file_a.md", "file_b.md"]
      }}
    ]

    RULES:
    - Each question MUST genuinely require facts from at least 2 different source_files listed above.
    - source_files must list ALL documents needed to answer the question.
    - reference_answer must be complete and cite the relevant facts.
    - keywords should be specific values an evaluator could look for in the answer.
    - Do NOT repeat questions already generated in single-document passes.
    - Holistic questions may reference many or all documents (e.g. counts, aggregations, comparisons across all).
    """


# ── Pass 3: Curation / Quality scoring ─────────────────────────────────────────

def pass3_curation_prompt(
    questions: list[dict],
    criteria: list[str],
    min_score: int,
    target_total: int,
) -> str:
    """
    Prompt for Pass 3: score and filter all candidate questions.
    """
    # TODO: check this parsing
    criteria_block = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))
    questions_block = json.dumps(questions, indent=2)

    return f"""You are a quality curator for an LLM evaluation dataset.

    You have {len(questions)} candidate questions. Your job is to:
    1. Score each question on a scale of 1-10 based on these criteria:
    {criteria_block}

    2. Select the best {target_total} questions that:
      - Score {min_score} or above
      - Are diverse (cover different facts, entities, and categories)
      - Include a healthy mix of single-doc and multi-doc questions

    CANDIDATE QUESTIONS:
    {questions_block}

    OUTPUT FORMAT — return a JSON array of the selected questions, each with a "quality_score" field added:
    [
      {{
        "question": "...",
        "keywords": [...],
        "reference_answer": "...",
        "category": "...",
        "source_files": [...],
        "quality_score": 8
      }}
    ]

    Return EXACTLY {target_total} questions (or fewer if not enough meet the quality bar).
    Order them from highest to lowest quality score.
    Do NOT include any questions scoring below {min_score}.
    """
