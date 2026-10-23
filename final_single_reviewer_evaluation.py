#!/usr/bin/env python3
"""
Final SkillScope retrieval evaluation using ONE human reviewer.

Use this only when a second independent reviewer is not available.

Inputs:
  results/reviewer_1_completed.csv
  results/hard_semantic_retrieval_test.json

Outputs:
  results/final_single_reviewer_retrieval_evaluation.json
  results/final_single_reviewer_retrieval_cases.csv

Method:
- Keep only cases Reviewer 1 marked YES.
- Use reviewer-approved acceptable chunk IDs as relevance labels.
- If acceptable_chunk_ids is blank on a YES row, fall back to source_chunk_id.
- Re-evaluate the saved rankings from the hard semantic experiment.
- Report TF-IDF, Embedding, and the three experimental hybrid variants.

Important limitation:
This is a single-reviewer human validation. It does not measure inter-rater
agreement and should not be described as independently double-reviewed.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

REVIEW_PATH = RESULTS_DIR / "reviewer_1_completed.csv"
HARD_RESULTS_PATH = RESULTS_DIR / "hard_semantic_retrieval_test.json"

OUT_JSON = RESULTS_DIR / "final_single_reviewer_retrieval_evaluation.json"
OUT_CSV = RESULTS_DIR / "final_single_reviewer_retrieval_cases.csv"

METHODS = [
    "TF-IDF",
    "Embedding",
    "Hybrid_75T_25E",
    "Hybrid_50T_50E",
    "Hybrid_25T_75E",
]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_ids(value: str) -> set[str]:
    return {
        item.strip()
        for item in (value or "").split(";")
        if item.strip()
    }


def first_relevant_rank(top5: list[str], acceptable: set[str]) -> int | None:
    for i, chunk_id in enumerate(top5, start=1):
        if chunk_id in acceptable:
            return i
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    hit1 = sum(r == 1 for r in ranks)
    hit3 = sum(r is not None and r <= 3 for r in ranks)
    hit5 = sum(r is not None and r <= 5 for r in ranks)
    mrr5 = sum((1.0 / r) if r is not None and r <= 5 else 0.0 for r in ranks)

    return {
        "cases": n,
        "hit_at_1": hit1 / n if n else 0.0,
        "hit_at_3": hit3 / n if n else 0.0,
        "hit_at_5": hit5 / n if n else 0.0,
        "mrr_at_5": mrr5 / n if n else 0.0,
        "miss_rate_at_5": (n - hit5) / n if n else 0.0,
        "misses_at_5": n - hit5,
    }


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def main() -> None:
    if not REVIEW_PATH.exists():
        raise SystemExit(f"Missing {REVIEW_PATH}")
    if not HARD_RESULTS_PATH.exists():
        raise SystemExit(f"Missing {HARD_RESULTS_PATH}")

    reviews = load_csv(REVIEW_PATH)
    hard = json.loads(HARD_RESULTS_PATH.read_text(encoding="utf-8"))
    hard_by_case = {row["case_id"]: row for row in hard["cases"]}

    approved = []
    excluded = []

    for row in reviews:
        decision = (row.get("semantic_equivalent_yes_no") or "").strip().upper()

        if decision == "YES":
            acceptable = split_ids(row.get("acceptable_chunk_ids", ""))
            if not acceptable:
                acceptable = {row["source_chunk_id"]}

            approved.append({
                "case_id": row["case_id"],
                "role": row["role"],
                "skill": row["skill"],
                "acceptable_chunk_ids": acceptable,
                "review_notes": row.get("notes", ""),
            })

        elif decision == "NO":
            excluded.append({
                "case_id": row["case_id"],
                "role": row["role"],
                "skill": row["skill"],
                "reason": row.get("notes", ""),
            })

        else:
            raise SystemExit(
                f"Incomplete reviewer decision for {row['case_id']}: {decision!r}"
            )

    if not approved:
        raise SystemExit("No reviewer-approved YES cases found.")

    method_ranks = {m: [] for m in METHODS}
    role_method_ranks = defaultdict(lambda: {m: [] for m in METHODS})
    case_rows = []

    for item in approved:
        case_id = item["case_id"]
        if case_id not in hard_by_case:
            raise SystemExit(f"{case_id} missing from hard semantic results.")

        source = hard_by_case[case_id]
        acceptable = item["acceptable_chunk_ids"]

        output_row = {
            "case_id": case_id,
            "role": item["role"],
            "skill": item["skill"],
            "acceptable_chunk_ids": ";".join(sorted(acceptable)),
            "review_notes": item["review_notes"],
        }

        for method in METHODS:
            top5 = source["methods"][method]["top5"]
            rank = first_relevant_rank(top5, acceptable)

            method_ranks[method].append(rank)
            role_method_ranks[item["role"]][method].append(rank)

            output_row[f"{method}_relevant_rank"] = "" if rank is None else rank
            output_row[f"{method}_hit_at_5"] = "NO" if rank is None else "YES"

        case_rows.append(output_row)

    overall = {method: metrics(method_ranks[method]) for method in METHODS}
    by_role = {
        role: {
            method: metrics(role_method_ranks[role][method])
            for method in METHODS
        }
        for role in sorted(role_method_ranks)
    }

    # Pairwise descriptive deltas versus the current TF-IDF baseline.
    baseline = overall["TF-IDF"]
    deltas = {}
    for method in METHODS:
        if method == "TF-IDF":
            continue
        m = overall[method]
        deltas[method] = {
            "hit_at_1_percentage_point_change_vs_tfidf":
                100 * (m["hit_at_1"] - baseline["hit_at_1"]),
            "hit_at_3_percentage_point_change_vs_tfidf":
                100 * (m["hit_at_3"] - baseline["hit_at_3"]),
            "hit_at_5_percentage_point_change_vs_tfidf":
                100 * (m["hit_at_5"] - baseline["hit_at_5"]),
            "mrr_at_5_change_vs_tfidf":
                m["mrr_at_5"] - baseline["mrr_at_5"],
            "top5_miss_reduction_count_vs_tfidf":
                baseline["misses_at_5"] - m["misses_at_5"],
        }

    role_counts = Counter(item["role"] for item in approved)

    report = {
        "evaluation_design": {
            "review_type": "single_human_reviewer",
            "reviewer_count": 1,
            "original_review_cases": len(reviews),
            "approved_cases_used_for_evaluation": len(approved),
            "excluded_cases": len(excluded),
            "approved_cases_by_role": dict(sorted(role_counts.items())),
            "limitation": (
                "A second independent reviewer was not available. "
                "Therefore inter-rater agreement and Cohen's kappa were not measured. "
                "Results should be described as single-reviewer human validation."
            ),
        },
        "overall_metrics": overall,
        "metrics_by_role": by_role,
        "changes_vs_tfidf": deltas,
        "excluded_cases": excluded,
    }

    RESULTS_DIR.mkdir(exist_ok=True)

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    print("=" * 82)
    print("SKILLSCOPE FINAL SINGLE-REVIEWER RETRIEVAL EVALUATION")
    print("=" * 82)
    print(f"Reviewed cases             : {len(reviews)}")
    print(f"Reviewer-approved YES cases: {len(approved)}")
    print(f"Excluded NO cases          : {len(excluded)}")
    print()
    print("Approved cases by role:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role:<28} {count}")

    print("\nOVERALL RESULTS")
    print("-" * 82)
    print(
        f"{'Method':<18}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'MRR@5':>10}"
        f"{'Miss@5':>10}"
        f"{'Misses':>8}"
    )

    for method in METHODS:
        m = overall[method]
        print(
            f"{method:<18}"
            f"{pct(m['hit_at_1']):>10}"
            f"{pct(m['hit_at_3']):>10}"
            f"{pct(m['hit_at_5']):>10}"
            f"{m['mrr_at_5']:>10.4f}"
            f"{pct(m['miss_rate_at_5']):>10}"
            f"{m['misses_at_5']:>8}"
        )

    print("\nCHANGES VS TF-IDF")
    print("-" * 82)
    for method, d in deltas.items():
        print(
            f"{method:<18} "
            f"Hit@5 {d['hit_at_5_percentage_point_change_vs_tfidf']:+.2f} pp | "
            f"MRR@5 {d['mrr_at_5_change_vs_tfidf']:+.4f} | "
            f"miss reduction {d['top5_miss_reduction_count_vs_tfidf']:+d}"
        )

    print("\nLIMITATION")
    print("-" * 82)
    print(
        "Only one human reviewer was available. "
        "Do not report inter-rater agreement or Cohen's kappa."
    )

    print("\nSaved:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_CSV}")


if __name__ == "__main__":
    main()
