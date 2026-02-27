"""
generator.py
------------
Orchestrates the three-pass dataset generation pipeline:

  Pass 1 — Per-file extraction + candidate question generation
  Pass 2 — Cross-document spanning/holistic question generation  
  Pass 3 — LLM-as-judge curation and quality filtering

Usage (from CLI via main.py, or imported directly):
  from generator import DatasetGenerator
  gen = DatasetGenerator(config)
  gen.run(kb_path="./knowledge_base", output_path="output_dataset.jsonl")
"""

# TODO: implement the weights from each categories in the config file

import json
import math
import logging
from pathlib import Path
from typing import Optional

from loaders import load_file, discover_files
from llm_client import call_llm, extract_json
from prompts import (
    SYSTEM_PROMPT,
    pass1_extraction_prompt,
    pass2_spanning_prompt,
    pass3_curation_prompt,
)

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, config: dict):
        # NOTE: CHECK
        self.cfg = config
        self.llm_cfg = config["llm"]
        self.dataset_cfg = config["dataset"]
        self.categories = config["categories"]
        self.curation_cfg = config["curation"]
        self.files_cfg = config["files"]
        self.paths_cfg = config["paths"]

        # Split categories into single-doc and multi-doc
        self.single_doc_cats = {
            k: v for k, v in self.categories.items() if not v.get("multi_doc", False)
        }
        self.multi_doc_cats = {
            k: v for k, v in self.categories.items() if v.get("multi_doc", False)
        }

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _llm(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        # NOTE: CHECK
        return call_llm(
            prompt=prompt,
            system=system,
            model=self.llm_cfg["model"],
            temperature=self.llm_cfg.get("temperature", 0.7),
            max_tokens=self.llm_cfg.get("max_tokens", 8192),
        )

    def _load_examples(self) -> list[dict]:
        # NOTE: CHECK
        examples_path = Path(self.paths_cfg["examples_file"])
        if not examples_path.exists():
            logger.warning(f"Examples file not found: {examples_path}. Proceeding without examples.")
            return []
        examples = []
        with open(examples_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSONL line in examples: {line[:60]}")
        logger.info(f"Loaded {len(examples)} examples from {examples_path}")
        return examples

    def _distribute_questions(self, files: list[Path]) -> dict[str, int]:
        """
        Distribute single-doc question budget across files proportionally by file size,
        respecting min_per_file and max_pct_per_file constraints.
        """
        # TODO: edge case: if total_questions is less than the min_per_file * number_of_files len(files)
        # TODO: add normalization to make sure its mathematically correct (carefull with rounded)
        total = self.dataset_cfg["total_questions"]
        spanning_ratio = self.dataset_cfg.get("spanning_ratio", 0.2)
        single_doc_budget = round(total * (1 - spanning_ratio))
        min_q = self.dataset_cfg.get("min_per_file", 1)
        max_q = math.ceil(total * self.dataset_cfg.get("max_pct_per_file", 0.2))

        # Allocate minimum to each file first
        allocation = {str(f): min_q for f in files}
        remaining = single_doc_budget - (min_q * len(files))

        if remaining <= 0:
            return allocation

        # Distribute remainder proportionally by file size
        sizes = {str(f): max(f.stat().st_size, 1) for f in files}
        total_size = sum(sizes.values())

        for f in files:
            key = str(f)
            proportional = round(remaining * sizes[key] / total_size)
            allocation[key] += proportional

        # Enforce ceiling
        for key in allocation:
            allocation[key] = min(allocation[key], max_q)

        return allocation

    def _save_intermediate(self, data: dict | list, name: str):
        """Save intermediate pass results to disk for debugging."""
        inter_dir = Path(self.paths_cfg.get("intermediate_dir", "intermediate/"))
        inter_dir.mkdir(parents=True, exist_ok=True)
        out_path = inter_dir / name
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved intermediate: {out_path}")

    # ── Pass 1 ──────────────────────────────────────────────────────────────────

    def pass1_extract_and_generate(
        self, files: list[Path], allocation: dict[str, int], examples: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        For each file: extract facts and generate candidate single-doc questions.
        Returns (all_extractions, all_single_doc_questions).
        """
        all_extractions = []
        all_single_doc_questions = []

        for i, file_path in enumerate(files):
            key = str(file_path)
            n_questions = allocation.get(key, 1)
            logger.info(f"[Pass 1] {i+1}/{len(files)}: {file_path.name} → {n_questions} questions")

            try:
                content = load_file(file_path)
            except Exception as e:
                logger.error(f"  Failed to load {file_path}: {e}")
                continue

            prompt = pass1_extraction_prompt(
                filename=file_path.name,
                content=content,
                single_doc_categories=self.single_doc_cats,
                examples=examples,
                n_questions=n_questions,
            )
            print(f"Prompt 1: {prompt}")

            try:
                raw = self._llm(prompt)
                result = extract_json(raw)

                # Validate structure
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object from Pass 1")

                extraction = {
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "extracted_facts": result.get("extracted_facts", []),
                }
                all_extractions.append(extraction)

                questions = result.get("candidate_questions", [])
                # Ensure source_files is always populated
                for q in questions:
                    if "source_files" not in q or not q["source_files"]:
                        q["source_files"] = [file_path.name]
                all_single_doc_questions.extend(questions)

                logger.info(f"  ✓ {len(questions)} questions, {len(extraction['extracted_facts'])} facts")

            except Exception as e:
                logger.error(f"  LLM or parse error for {file_path.name}: {e}")
                continue

        self._save_intermediate(all_extractions, "pass1_extractions.json")
        self._save_intermediate(all_single_doc_questions, "pass1_questions.json")
        return all_extractions, all_single_doc_questions

    # ── Pass 2 ──────────────────────────────────────────────────────────────────

    def pass2_spanning(
        self, all_extractions: list[dict], examples: list[dict]
    ) -> list[dict]:
        """
        Generate spanning/holistic questions from combined extracted facts.
        """
        if not self.multi_doc_cats:
            logger.info("[Pass 2] No multi-doc categories configured. Skipping.")
            return []

        total = self.dataset_cfg["total_questions"]
        n_spanning = round(total * self.dataset_cfg.get("spanning_ratio", 0.2))

        if len(all_extractions) < 2:
            logger.warning("[Pass 2] Need at least 2 documents for spanning questions. Skipping.")
            return []

        logger.info(f"[Pass 2] Generating {n_spanning} spanning/holistic questions across {len(all_extractions)} documents")

        prompt = pass2_spanning_prompt(
            all_facts=all_extractions,
            multi_doc_categories=self.multi_doc_cats,
            examples=examples,
            n_spanning=n_spanning,
        )

        try:
            raw = self._llm(prompt)
            questions = extract_json(raw)
            if not isinstance(questions, list):
                raise ValueError("Expected a JSON array from Pass 2")
            logger.info(f"  ✓ {len(questions)} spanning questions generated")
        except Exception as e:
            logger.error(f"  Pass 2 error: {e}")
            questions = []

        self._save_intermediate(questions, "pass2_questions.json")
        return questions

    # ── Pass 3 ──────────────────────────────────────────────────────────────────

    def pass3_curate(self, all_questions: list[dict]) -> list[dict]:
        """
        LLM-as-judge: score and filter all candidate questions down to target total.
        If curation is disabled, return the top N by naive ordering.
        """
        total = self.dataset_cfg["total_questions"]
        logger.info(f"[Pass 3] Curating {len(all_questions)} candidates → {total} final questions")

        if not self.curation_cfg.get("enabled", True):
            logger.info("  Curation disabled. Taking first N questions.")
            return all_questions[:total]

        # If we have too many candidates to send in one call, batch them
        # LLMs can handle ~100-200 questions comfortably in one call
        batch_size = 80
        if len(all_questions) > batch_size:
            # Do a pre-filter: score in batches, then do a final selection pass
            scored = []
            for i in range(0, len(all_questions), batch_size):
                batch = all_questions[i:i + batch_size]
                scored.extend(self._curate_batch(batch, total=len(batch)))
            # Now do a final selection from scored pool
            scored.sort(key=lambda q: q.get("quality_score", 0), reverse=True)
            return scored[:total]
        else:
            return self._curate_batch(all_questions, total=total)

    def _curate_batch(self, questions: list[dict], total: int) -> list[dict]:
        min_score = self.curation_cfg.get("min_score", 7)
        criteria = self.curation_cfg.get("criteria", [])

        prompt = pass3_curation_prompt(
            questions=questions,
            criteria=criteria,
            min_score=min_score,
            target_total=total,
        )

        try:
            raw = self._llm(prompt)
            curated = extract_json(raw)
            if not isinstance(curated, list):
                raise ValueError("Expected a JSON array from Pass 3")
            logger.info(f"  ✓ {len(curated)} questions passed curation")
            return curated
        except Exception as e:
            logger.error(f"  Pass 3 error: {e}. Returning uncurated questions.")
            return questions[:total]

    # ── Main entrypoint ─────────────────────────────────────────────────────────

    def run(self, kb_path: str, output_path: Optional[str] = None):
        """
        Full pipeline: discover files → Pass 1 → Pass 2 → Pass 3 → write JSONL.
        """
        output_path = output_path or self.paths_cfg.get("output_file", "test-sets/generated/output_dataset.jsonl")

        logger.info("=" * 60)
        logger.info("  Eval Dataset Generator — Starting")
        logger.info("=" * 60)

        # Discover files
        supported = self.files_cfg.get("supported_extensions", [".md", ".txt"])
        files = discover_files(kb_path, supported)
        if not files:
            raise ValueError(f"No supported files found in: {kb_path}")
        logger.info(f"Discovered {len(files)} files in '{kb_path}'")

        # Load examples
        examples = self._load_examples()

        # Distribute question budget
        allocation = self._distribute_questions(files)
        logger.info(f"Question allocation: {allocation}")

        # TODO: CHECK
        # ── Pass 1 ────────────────────────────────────────────────────────────
        logger.info("\n── PASS 1: Extraction & Single-Doc Generation ─────────────")
        extractions, single_doc_qs = self.pass1_extract_and_generate(files, allocation, examples)

        # ── Pass 2 ────────────────────────────────────────────────────────────
        logger.info("\n── PASS 2: Spanning & Holistic Generation ─────────────────")
        spanning_qs = self.pass2_spanning(extractions, examples)

        # Combine all candidates
        all_candidates = single_doc_qs + spanning_qs
        logger.info(f"\nTotal candidates before curation: {len(all_candidates)}")
        logger.info(f"  Single-doc: {len(single_doc_qs)}")
        logger.info(f"  Spanning/holistic: {len(spanning_qs)}")

        # ── Pass 3 ────────────────────────────────────────────────────────────
        logger.info("\n── PASS 3: Curation & Quality Filtering ───────────────────")
        final_questions = self.pass3_curate(all_candidates)

        # ── Write output ──────────────────────────────────────────────────────
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for q in final_questions:
                # Ensure consistent field order in output
                record = {
                    "question": q.get("question", ""),
                    "keywords": q.get("keywords", []),
                    "reference_answer": q.get("reference_answer", ""),
                    "category": q.get("category", ""),
                    "source_files": q.get("source_files", []),
                    "quality_score": q.get("quality_score"),
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"\n✅ Done! {len(final_questions)} questions written to: {out_path}")
        logger.info("=" * 60)

        return final_questions
