#!/usr/bin/env python3
"""
Build a balanced, human-review held-out set for SkillScope retrieval evaluation.

Reads:
  results/hard_semantic_retrieval_test.json
  data/source/knowledge_base.jsonl

Writes:
  results/human_review_heldout_blinded.csv
  results/human_review_heldout_selection_audit.json

Selection:
- 30 total cases = 6 per role
- prioritises cases where Embedding improves over TF-IDF
- then fills remaining slots with low-overlap / difficult cases
- reviewers do NOT see retriever ranks in the CSV (blinded review)

Reviewer task:
1. Decide whether the query preserves the intended meaning of the source chunk.
2. If YES, list all acceptable chunk IDs that could correctly answer the query.
3. If NO, briefly explain the semantic drift.
4. Two reviewers complete their columns independently.
5. Resolve disagreements in the adjudicated columns.

This script does not modify the application or retrieval model.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
KB_PATH = BASE_DIR / "data" / "source" / "knowledge_base.jsonl"
HARD_RESULTS_PATH = RESULTS_DIR / "hard_semantic_retrieval_test.json"

OUT_CSV = RESULTS_DIR / "human_review_heldout_blinded.csv"
OUT_AUDIT = RESULTS_DIR / "human_review_heldout_selection_audit.json"

CASES_PER_ROLE = 6


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rank(row: dict, method: str) -> int:
    return int(row["methods"][method]["rank"])


def selection_score(row: dict) -> tuple:
    tf = rank(row, "TF-IDF")
    em = rank(row, "Embedding")
    overlap = float(row.get("content_word_overlap", 1.0))

    tf_miss = int(tf > 5)
    em_hit = int(em <= 5)
    embedding_gain = tf - em

    return (
        tf_miss * em_hit,
        embedding_gain,
        -overlap,
        tf,
    )


def main() -> None:
    if not HARD_RESULTS_PATH.exists():
        raise SystemExit(
            f"Missing {HARD_RESULTS_PATH}\n"
            "Run hard_semantic_retrieval_test.py first."
        )

    if not KB_PATH.exists():
        raise SystemExit(f"Missing {KB_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)

    report = json.loads(HARD_RESULTS_PATH.read_text(encoding="utf-8"))
    kb = load_jsonl(KB_PATH)
    chunk_by_id = {c["chunk_id"]: c for c in kb}

    cases = report.get("cases", [])
    by_role = defaultdict(list)
    for row in cases:
        by_role[row["role"]].append(row)

    selected = []
    for role in sorted(by_role):
        ranked = sorted(by_role[role], key=selection_score, reverse=True)
        selected.extend(ranked[:CASES_PER_ROLE])

    selected.sort(key=lambda x: (x["role"], x["case_id"]))

    review_rows = []
    audit_rows = []

    for row in selected:
        expected = row["expected_chunk_id"]
        source = chunk_by_id.get(expected)
        if source is None:
            raise SystemExit(f"Expected chunk {expected} not found in knowledge base.")

        review_rows.append({
            "case_id": row["case_id"],
            "role": row["role"],
            "skill": row.get("skill", ""),
            "candidate_query": row["query"],
            "source_chunk_id": expected,
            "source_chunk_text": source.get("text", ""),
            "source_title": source.get("source_title", source.get("source", "")),
            "reviewer_1_semantic_equivalent_yes_no": "",
            "reviewer_1_acceptable_chunk_ids": "",
            "reviewer_1_notes": "",
            "reviewer_2_semantic_equivalent_yes_no": "",
            "reviewer_2_acceptable_chunk_ids": "",
            "reviewer_2_notes": "",
            "adjudicated_semantic_equivalent_yes_no": "",
            "adjudicated_acceptable_chunk_ids": "",
            "adjudication_notes": "",
        })

        audit_rows.append({
            "case_id": row["case_id"],
            "role": row["role"],
            "skill": row.get("skill", ""),
            "source_chunk_id": expected,
            "content_word_overlap": row.get("content_word_overlap"),
            "tfidf_rank": rank(row, "TF-IDF"),
            "embedding_rank": rank(row, "Embedding"),
            "hybrid_75T_25E_rank": rank(row, "Hybrid_75T_25E"),
            "hybrid_50T_50E_rank": rank(row, "Hybrid_50T_50E"),
            "hybrid_25T_75E_rank": rank(row, "Hybrid_25T_75E"),
            "selection_score": list(selection_score(row)),
        })

    fieldnames = list(review_rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    OUT_AUDIT.write_text(
        json.dumps({
            "selection_method": (
                "30 balanced cases (6 per role), prioritising TF-IDF misses "
                "recovered by embedding, then larger embedding rank gains "
                "and low lexical overlap."
            ),
            "reviewer_blinding": (
                "Retriever ranks are intentionally excluded from the CSV."
            ),
            "case_count": len(review_rows),
            "cases_per_role": CASES_PER_ROLE,
            "cases": audit_rows,
        }, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("SKILLSCOPE HUMAN-REVIEW HELD-OUT SET")
    print("=" * 72)
    print(f"Selected cases : {len(review_rows)}")
    print(f"Cases per role : {CASES_PER_ROLE}")
    print()
    print("Blinded reviewer worksheet:")
    print(f"  {OUT_CSV}")
    print()
    print("Selection audit (do not give to reviewers):")
    print(f"  {OUT_AUDIT}")
    print()
    print("Reviewer instructions:")
    print("  1. Read candidate_query and source_chunk_text.")
    print("  2. Mark semantic equivalent: YES or NO.")
    print("  3. If YES, list ALL acceptable chunk IDs.")
    print("  4. Add a short note where needed.")
    print("  5. Reviewers work independently.")
    print("  6. Resolve disagreements in adjudicated_* columns.")
    print()
    print("Do not evaluate TF-IDF/Embedding performance until adjudication is complete.")


if __name__ == "__main__":
    main()
