#!/usr/bin/env python3
"""
Merge two independent SkillScope reviewer files and prepare adjudication.

Expected inputs:
  results/reviewer_1_completed.csv
  results/reviewer_2_completed.csv

Outputs:
  results/reviewer_agreement_summary.json
  results/reviewer_adjudication.csv

The adjudication CSV contains both reviewers' decisions side by side and
blank adjudicated_* columns to be resolved only where necessary.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

R1_PATH = RESULTS_DIR / "reviewer_1_completed.csv"
R2_PATH = RESULTS_DIR / "reviewer_2_completed.csv"

OUT_CSV = RESULTS_DIR / "reviewer_adjudication.csv"
OUT_JSON = RESULTS_DIR / "reviewer_agreement_summary.json"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_yes_no(value: str) -> str:
    v = (value or "").strip().upper()
    return v if v in {"YES", "NO"} else ""


def split_ids(value: str) -> set[str]:
    return {
        x.strip()
        for x in (value or "").split(";")
        if x.strip()
    }


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or len(a) != len(b):
        return None

    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n

    ca = Counter(a)
    cb = Counter(b)

    expected = 0.0
    for label in {"YES", "NO"}:
        expected += (ca[label] / n) * (cb[label] / n)

    if expected == 1.0:
        return 1.0

    return (observed - expected) / (1.0 - expected)


def main() -> None:
    if not R1_PATH.exists():
        raise SystemExit(f"Missing {R1_PATH}")
    if not R2_PATH.exists():
        raise SystemExit(
            f"Missing {R2_PATH}\n"
            "Reviewer 2 must complete reviewer_2_packet.csv first and save it "
            "as reviewer_2_completed.csv in results/."
        )

    r1 = load_csv(R1_PATH)
    r2 = load_csv(R2_PATH)

    if len(r1) != len(r2):
        raise SystemExit(
            f"Reviewer files have different lengths: {len(r1)} vs {len(r2)}"
        )

    r2_by_id = {row["case_id"]: row for row in r2}

    merged = []
    labels1 = []
    labels2 = []

    decision_agreements = 0
    chunk_agreements_on_yes = 0
    both_yes_count = 0
    disagreements = []

    for row1 in r1:
        case_id = row1["case_id"]
        if case_id not in r2_by_id:
            raise SystemExit(f"Reviewer 2 is missing case {case_id}")

        row2 = r2_by_id[case_id]

        d1 = norm_yes_no(row1.get("semantic_equivalent_yes_no", ""))
        d2 = norm_yes_no(row2.get("semantic_equivalent_yes_no", ""))

        if not d1 or not d2:
            raise SystemExit(
                f"Incomplete YES/NO review for {case_id}: "
                f"reviewer1={d1!r}, reviewer2={d2!r}"
            )

        labels1.append(d1)
        labels2.append(d2)

        same_decision = d1 == d2
        if same_decision:
            decision_agreements += 1

        ids1 = split_ids(row1.get("acceptable_chunk_ids", ""))
        ids2 = split_ids(row2.get("acceptable_chunk_ids", ""))

        chunk_ids_agree = None
        if d1 == "YES" and d2 == "YES":
            both_yes_count += 1
            chunk_ids_agree = ids1 == ids2
            if chunk_ids_agree:
                chunk_agreements_on_yes += 1

        needs_adjudication = (
            not same_decision
            or (d1 == "YES" and d2 == "YES" and ids1 != ids2)
        )

        if needs_adjudication:
            disagreements.append(case_id)

        merged.append({
            "case_id": case_id,
            "role": row1.get("role", ""),
            "skill": row1.get("skill", ""),
            "candidate_query": row1.get("candidate_query", ""),
            "source_chunk_id": row1.get("source_chunk_id", ""),
            "source_chunk_text": row1.get("source_chunk_text", ""),
            "source_title": row1.get("source_title", ""),

            "reviewer_1_semantic_equivalent_yes_no": d1,
            "reviewer_1_acceptable_chunk_ids": row1.get("acceptable_chunk_ids", ""),
            "reviewer_1_notes": row1.get("notes", ""),

            "reviewer_2_semantic_equivalent_yes_no": d2,
            "reviewer_2_acceptable_chunk_ids": row2.get("acceptable_chunk_ids", ""),
            "reviewer_2_notes": row2.get("notes", ""),

            "decision_agreement": "YES" if same_decision else "NO",
            "acceptable_ids_agreement_when_both_yes": (
                "" if chunk_ids_agree is None
                else ("YES" if chunk_ids_agree else "NO")
            ),
            "needs_adjudication": "YES" if needs_adjudication else "NO",

            "adjudicated_semantic_equivalent_yes_no": (
                d1 if same_decision else ""
            ),
            "adjudicated_acceptable_chunk_ids": (
                row1.get("acceptable_chunk_ids", "")
                if same_decision and d1 == "YES" and ids1 == ids2
                else ""
            ),
            "adjudication_notes": "",
        })

    fieldnames = list(merged[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    n = len(merged)
    agreement_rate = decision_agreements / n
    kappa = cohen_kappa(labels1, labels2)
    chunk_agreement_rate = (
        chunk_agreements_on_yes / both_yes_count
        if both_yes_count
        else None
    )

    summary = {
        "case_count": n,
        "reviewer_1_counts": dict(Counter(labels1)),
        "reviewer_2_counts": dict(Counter(labels2)),
        "decision_agreement_count": decision_agreements,
        "decision_agreement_rate": agreement_rate,
        "cohen_kappa": kappa,
        "both_yes_count": both_yes_count,
        "acceptable_chunk_id_exact_agreement_count": chunk_agreements_on_yes,
        "acceptable_chunk_id_exact_agreement_rate_when_both_yes": chunk_agreement_rate,
        "cases_needing_adjudication": disagreements,
        "cases_needing_adjudication_count": len(disagreements),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("SKILLSCOPE REVIEWER AGREEMENT")
    print("=" * 72)
    print(f"Cases                         : {n}")
    print(f"Decision agreement            : {decision_agreements}/{n} "
          f"({agreement_rate:.2%})")
    print(f"Cohen's kappa                 : "
          f"{'N/A' if kappa is None else f'{kappa:.4f}'}")

    if chunk_agreement_rate is not None:
        print(
            "Acceptable-ID agreement (both YES): "
            f"{chunk_agreements_on_yes}/{both_yes_count} "
            f"({chunk_agreement_rate:.2%})"
        )

    print(f"Cases needing adjudication    : {len(disagreements)}")
    for case_id in disagreements:
        print(f"  - {case_id}")

    print("\nAdjudication worksheet:")
    print(f"  {OUT_CSV}")

    print("\nAgreement summary:")
    print(f"  {OUT_JSON}")

    print("\nNEXT:")
    print("Resolve only the rows marked needs_adjudication=YES.")
    print("After that, run the final retrieval evaluation on adjudicated YES cases.")


if __name__ == "__main__":
    main()
