"""Run one full SkillScope RAG evaluation through local Ollama."""

from __future__ import annotations

import argparse
import json

from rag_pipeline import RAGEvaluator
from skillscope_core import RoleAwareRetriever, load_project_data


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillScope AI RAG evaluation demo")
    parser.add_argument("--question-id", default="Q_QA_T_01")
    parser.add_argument("--answer", required=True)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    chunks, questions, _ = load_project_data()
    question_by_id = {row["question_id"]: row for row in questions}
    if args.question_id not in question_by_id:
        raise SystemExit(f"Unknown question ID: {args.question_id}")

    evaluator = RAGEvaluator(RoleAwareRetriever(chunks))
    if args.fallback:
        result = evaluator.evaluate_with_safe_fallback(question_by_id[args.question_id], args.answer)
    else:
        result = evaluator.evaluate(question_by_id[args.question_id], args.answer)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
