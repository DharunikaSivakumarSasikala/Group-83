"""Generate the 25 deterministic assessment set blueprints."""

import json
from pathlib import Path

from skillscope_core import build_assessment_sets, load_project_data, load_quizzes


def main() -> None:
    _, questions, _ = load_project_data()
    sets = build_assessment_sets(questions, load_quizzes())
    output = Path(__file__).resolve().parent / "data" / "assessment_sets.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(sets, indent=2), encoding="utf-8")
    print(f"Created {sum(map(len, sets.values()))} sets with 7 written and 3 quiz questions each: {output}")


if __name__ == "__main__":
    main()
