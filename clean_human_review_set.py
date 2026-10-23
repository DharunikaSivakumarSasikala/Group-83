#!/usr/bin/env python3
"""
Clean the SkillScope 30-case human-review held-out set before reviewer labelling.

This script:
- removes seven cases with obvious semantic drift,
- replaces them with safer cases from the 60-case hard semantic experiment,
- strips LLM meta-commentary such as
  "Here's a challenging semantic retrieval query: ..."
  while keeping the actual query,
- preserves 30 total cases and 6 cases per role,
- keeps retriever ranks OUT of the reviewer CSV,
- writes an audit file separately.

Inputs:
  results/hard_semantic_retrieval_test.json
  data/source/knowledge_base.jsonl

Outputs:
  results/human_review_heldout_cleaned.csv
  results/human_review_heldout_cleaned_audit.json
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
HARD_RESULTS = RESULTS_DIR / "hard_semantic_retrieval_test.json"
KB_PATH = BASE_DIR / "data" / "source" / "knowledge_base.jsonl"

OUT_CSV = RESULTS_DIR / "human_review_heldout_cleaned.csv"
OUT_AUDIT = RESULTS_DIR / "human_review_heldout_cleaned_audit.json"

# Cases previously selected but unsuitable because the generated query
# materially changes, contradicts, or over-generalises the source concept.
REMOVE_CASES = {
    "HARD_BACK_019",   # loses signed overflow / narrowing conversion meaning
    "HARD_BACK_033",   # too vague for implicit Go interface satisfaction
    "HARD_BACK_040",   # over-generalises playback resilience controls
    "HARD_DEVO_004",   # incorrectly implies ConfigMaps hold sensitive data
    "HARD_DEVO_008",   # conflates Dockerfile and Compose purposes
    "HARD_FRON_008",   # changes contrast-ratio requirement into font separation
    "HARD_QA_T_021",   # too broad; loses risk-based device-matrix meaning
}

# Safer replacements from the same 60-case experiment.
REPLACEMENTS = {
    "backend_software_engineer": [
        "HARD_BACK_001",
        "HARD_BACK_008",
        "HARD_BACK_012",
    ],
    "devops_sre_engineer": [
        "HARD_DEVO_018",
        "HARD_DEVO_020",
    ],
    "frontend_ui_developer": [
        "HARD_FRON_004",
    ],
    "qa_test_engineer": [
        "HARD_QA_T_005",
    ],
}

# Original 30 selected case IDs, grouped implicitly by the prior audit.
ORIGINAL_SELECTED = [
    "HARD_BACK_015",
    "HARD_BACK_019",
    "HARD_BACK_022",
    "HARD_BACK_026",
    "HARD_BACK_033",
    "HARD_BACK_040",

    "HARD_DATA_001",
    "HARD_DATA_006",
    "HARD_DATA_010",
    "HARD_DATA_015",
    "HARD_DATA_017",
    "HARD_DATA_020",

    "HARD_DEVO_001",
    "HARD_DEVO_004",
    "HARD_DEVO_008",
    "HARD_DEVO_010",
    "HARD_DEVO_013",
    "HARD_DEVO_017",

    "HARD_FRON_008",
    "HARD_FRON_009",
    "HARD_FRON_011",
    "HARD_FRON_014",
    "HARD_FRON_016",
    "HARD_FRON_019",

    "HARD_QA_T_006",
    "HARD_QA_T_008",
    "HARD_QA_T_010",
    "HARD_QA_T_014",
    "HARD_QA_T_019",
    "HARD_QA_T_021",
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_query(text: str) -> str:
    """Remove common LLM meta-commentary while preserving the actual query."""
    text = " ".join((text or "").split())

    # Most generated meta responses contain the intended query in quotes.
    if text.lower().startswith(("here's a", "here is a")):
        quoted = re.search(r'"([^"]+)"', text)
        if quoted:
            return quoted.group(1).strip()

    # Remove simple leading labels if present.
    text = re.sub(
        r"^(?:question|query|paraphrase)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def rank(case: dict, method: str) -> int:
    return int(case["methods"][method]["rank"])


def main() -> None:
    if not HARD_RESULTS.exists():
        raise SystemExit(
            f"Missing {HARD_RESULTS}\n"
            "Run hard_semantic_retrieval_test.py first."
        )
    if not KB_PATH.exists():
        raise SystemExit(f"Missing {KB_PATH}")

    report = json.loads(HARD_RESULTS.read_text(encoding="utf-8"))
    all_cases = {row["case_id"]: row for row in report["cases"]}

    kb = load_jsonl(KB_PATH)
    kb_by_id = {row["chunk_id"]: row for row in kb}

    # Start with original 30 minus drift cases.
    final_ids = [
        case_id
        for case_id in ORIGINAL_SELECTED
        if case_id not in REMOVE_CASES
    ]

    # Add balanced replacements.
    for role, case_ids in REPLACEMENTS.items():
        final_ids.extend(case_ids)

    if len(final_ids) != 30:
        raise SystemExit(f"Expected 30 cases, found {len(final_ids)}")

    # Ensure there are no accidental duplicates.
    if len(set(final_ids)) != len(final_ids):
        raise SystemExit("Duplicate case IDs detected.")

    final_cases = []
    for case_id in final_ids:
        if case_id not in all_cases:
            raise SystemExit(f"Case not found in hard results: {case_id}")
        final_cases.append(all_cases[case_id])

    # Verify role balance.
    role_counts = Counter(row["role"] for row in final_cases)
    if any(count != 6 for count in role_counts.values()) or len(role_counts) != 5:
        raise SystemExit(f"Role balance is incorrect: {dict(role_counts)}")

    # Sort by role then case ID for an easy worksheet.
    final_cases.sort(key=lambda row: (row["role"], row["case_id"]))

    review_rows = []
    audit_rows = []

    for row in final_cases:
        chunk_id = row["expected_chunk_id"]
        source = kb_by_id.get(chunk_id)
        if source is None:
            raise SystemExit(f"Knowledge chunk not found: {chunk_id}")

        cleaned = clean_query(row["query"])

        review_rows.append({
            "case_id": row["case_id"],
            "role": row["role"],
            "skill": row.get("skill", ""),
            "candidate_query": cleaned,
            "source_chunk_id": chunk_id,
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
            "source_chunk_id": chunk_id,
            "was_replacement": row["case_id"] not in ORIGINAL_SELECTED,
            "content_word_overlap": row.get("content_word_overlap"),
            "tfidf_rank": rank(row, "TF-IDF"),
            "embedding_rank": rank(row, "Embedding"),
            "hybrid_75T_25E_rank": rank(row, "Hybrid_75T_25E"),
            "hybrid_50T_50E_rank": rank(row, "Hybrid_50T_50E"),
            "hybrid_25T_75E_rank": rank(row, "Hybrid_25T_75E"),
        })

    RESULTS_DIR.mkdir(exist_ok=True)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    OUT_AUDIT.write_text(
        json.dumps({
            "case_count": len(review_rows),
            "cases_per_role": 6,
            "role_counts": dict(sorted(role_counts.items())),
            "removed_for_semantic_drift": sorted(REMOVE_CASES),
            "replacement_case_ids": [
                case_id
                for ids in REPLACEMENTS.values()
                for case_id in ids
            ],
            "reviewer_blinding": (
                "Retriever ranks are not included in the reviewer CSV. "
                "They appear only in this audit file."
            ),
            "cases": audit_rows,
        }, indent=2),
        encoding="utf-8",
    )

    print("=" * 74)
    print("SKILLSCOPE CLEAN HUMAN-REVIEW SET")
    print("=" * 74)
    print(f"Total cases : {len(review_rows)}")
    print("Role counts :")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")

    print("\nRemoved drift cases:")
    for case_id in sorted(REMOVE_CASES):
        print(f"  - {case_id}")

    print("\nAdded replacements:")
    for role, ids in REPLACEMENTS.items():
        for case_id in ids:
            print(f"  + {case_id} ({role})")

    print("\nReviewer worksheet:")
    print(f"  {OUT_CSV}")

    print("\nAudit file (keep away from reviewers):")
    print(f"  {OUT_AUDIT}")

    print("\nNEXT:")
    print("Give the cleaned CSV to Reviewer 1 and Reviewer 2 independently.")
    print("Do not show them the audit file or retrieval ranks.")


if __name__ == "__main__":
    main()
