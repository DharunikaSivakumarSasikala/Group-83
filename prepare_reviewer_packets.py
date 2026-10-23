#!/usr/bin/env python3
"""
Prepare two independent reviewer packets from the cleaned SkillScope held-out set.

Input:
  results/human_review_heldout_cleaned.csv

Outputs:
  results/reviewer_1_packet.csv
  results/reviewer_2_packet.csv
  results/reviewer_instructions.txt

The reviewer packets intentionally exclude:
- TF-IDF ranks
- embedding ranks
- hybrid ranks
- the other reviewer's decisions

This preserves independent, blinded review.
"""

from __future__ import annotations

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_CSV = RESULTS_DIR / "human_review_heldout_cleaned.csv"

OUT_R1 = RESULTS_DIR / "reviewer_1_packet.csv"
OUT_R2 = RESULTS_DIR / "reviewer_2_packet.csv"
OUT_INSTRUCTIONS = RESULTS_DIR / "reviewer_instructions.txt"

BASE_FIELDS = [
    "case_id",
    "role",
    "skill",
    "candidate_query",
    "source_chunk_id",
    "source_chunk_text",
    "source_title",
]

REVIEW_FIELDS = [
    "semantic_equivalent_yes_no",
    "acceptable_chunk_ids",
    "notes",
]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_packet(path: Path, rows: list[dict]) -> None:
    fieldnames = BASE_FIELDS + REVIEW_FIELDS

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                **{field: row.get(field, "") for field in BASE_FIELDS},
                "semantic_equivalent_yes_no": "",
                "acceptable_chunk_ids": "",
                "notes": "",
            })


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(
            f"Missing {INPUT_CSV}\n"
            "Run clean_human_review_set.py first."
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = load_rows(INPUT_CSV)

    if len(rows) != 30:
        raise SystemExit(
            f"Expected 30 cleaned cases but found {len(rows)}."
        )

    write_packet(OUT_R1, rows)
    write_packet(OUT_R2, rows)

    instructions = """SKILLSCOPE RETRIEVAL HELD-OUT REVIEW

Purpose
-------
Check whether each paraphrased query preserves the technical meaning of its
source knowledge chunk. Do not judge which retrieval model is better.

Review each case independently.

Column: semantic_equivalent_yes_no
----------------------------------
Enter YES when the candidate query still asks about substantially the same
technical concept as the source chunk.

Enter NO when the query:
- changes an important technical meaning,
- contradicts the source,
- removes so much detail that a different concept could reasonably answer it,
- or introduces requirements not supported by the source.

Column: acceptable_chunk_ids
----------------------------
If semantic_equivalent_yes_no = YES:
- enter the source_chunk_id at minimum when it is appropriate evidence;
- if another knowledge chunk would ALSO be a valid answer, include all
  acceptable IDs separated by semicolons.

Example:
DATA_004;DATA_003

If semantic_equivalent_yes_no = NO:
leave acceptable_chunk_ids blank.

Column: notes
-------------
Briefly explain NO decisions or ambiguous cases.
For obvious YES cases, notes may be left blank.

Important
---------
1. Reviewers must work independently.
2. Do not look at TF-IDF, embedding, or hybrid retrieval ranks.
3. Do not copy the other reviewer's answers.
4. Judge semantic validity, not writing style.
5. It is acceptable to mark a case NO if it became too vague.

After both reviewers finish, return both completed CSV files for adjudication
and final TF-IDF-vs-embedding evaluation.
"""

    OUT_INSTRUCTIONS.write_text(instructions, encoding="utf-8")

    print("=" * 72)
    print("SKILLSCOPE REVIEWER PACKETS")
    print("=" * 72)
    print(f"Cases per reviewer : {len(rows)}")
    print()
    print("Reviewer 1:")
    print(f"  {OUT_R1}")
    print()
    print("Reviewer 2:")
    print(f"  {OUT_R2}")
    print()
    print("Instructions:")
    print(f"  {OUT_INSTRUCTIONS}")
    print()
    print("NEXT:")
    print("Send Reviewer 1 and Reviewer 2 their own CSV files.")
    print("They should complete them independently and return both files.")


if __name__ == "__main__":
    main()
