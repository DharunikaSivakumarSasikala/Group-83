"""Calculate human-labelled RAG hallucination and scoring metrics.

Input JSONL fields per case:
human_score, model_score, cited_chunk_ids, allowed_chunk_ids,
claim_support (array of booleans), expected_citation_ids (optional).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def calculate_metrics(rows: list[dict], competency_threshold: float = 80.0) -> dict:
    if not rows:
        raise ValueError("At least one labelled evaluation case is required.")
    errors, agreements = [], []
    valid_citations = total_citations = expected_found = expected_total = 0
    supported_claims = total_claims = 0
    for row in rows:
        human, model = float(row["human_score"]), float(row["model_score"])
        errors.append(abs(human - model))
        agreements.append((human >= competency_threshold) == (model >= competency_threshold))
        cited = set(row.get("cited_chunk_ids", []))
        allowed = set(row.get("allowed_chunk_ids", []))
        expected = set(row.get("expected_citation_ids", []))
        valid_citations += len(cited & allowed)
        total_citations += len(cited)
        expected_found += len(cited & expected)
        expected_total += len(expected)
        support = [bool(x) for x in row.get("claim_support", [])]
        supported_claims += sum(support)
        total_claims += len(support)
    groundedness = supported_claims / total_claims if total_claims else 0.0
    return {
        "cases": len(rows),
        "score_mae": round(sum(errors) / len(errors), 4),
        "competency_agreement": round(sum(agreements) / len(agreements), 4),
        "citation_precision": round(valid_citations / total_citations, 4) if total_citations else 0.0,
        "citation_recall": round(expected_found / expected_total, 4) if expected_total else 0.0,
        "claim_groundedness": round(groundedness, 4),
        "hallucination_rate": round(1.0 - groundedness, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Human-labelled JSONL file")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.path).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(calculate_metrics(rows), indent=2))


if __name__ == "__main__":
    main()
