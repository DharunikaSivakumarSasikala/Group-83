"""Interactive 10-question SkillScope AI RAG assessment demo.

Flow
----
1. Ask for an anonymous participant ID.
2. Ask the participant to select a technical course/domain.
3. Deterministically assign one of five 10-question sets for that domain.
4. Ask the 10 questions one at a time.
5. Accept normal answers, N/A, I don't know, SKIP, or a blank line.
6. After submission, evaluate answered questions with the real RAG pipeline:
   role-aware TF-IDF retrieval -> augmented prompt -> local Ollama generation
   -> validation.
7. Give an overall score, knowledge-gap summary, and trusted reading links.

This is a mentor/demo interface over the existing SkillScope implementation.
It reuses skillscope_core.py and rag_pipeline.py; it does not replace them.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from typing import Iterable

from rag_pipeline import (
    GenerationError,
    OllamaClient,
    SYSTEM_PROMPT,
    build_augmented_prompt,
    validate_generation,
)
from skillscope_core import (
    RoleAwareRetriever,
    assign_set,
    build_assessment_sets,
    load_project_data,
    score_answer,
)


WIDTH = 82
TOP_K_DEFAULT = 5

ROLE_OPTIONS = [
    ("backend_software_engineer", "Backend Software Engineering"),
    ("frontend_ui_developer", "Frontend / UI Development"),
    ("devops_sre_engineer", "DevOps / SRE"),
    ("qa_test_engineer", "QA / Testing"),
    ("data_engineer", "Data Engineering"),
]
ROLE_LABELS = dict(ROLE_OPTIONS)

SKIP_ANSWERS = {
    "",
    "skip",
    "skipped",
    "n/a",
    "na",
    "not applicable",
    "i don't know",
    "i dont know",
    "i do not know",
    "don't know",
    "dont know",
    "do not know",
}


@dataclass
class QuestionResult:
    question: dict
    answer: str
    status: str
    score: float
    max_score: float
    strengths: list[str]
    missing_concepts: list[str]
    improvement_areas: list[dict]
    confidence: str
    rationale: str
    evidence_chunk_ids: list[str]
    retrieved_evidence: list[dict]
    mode: str


def line(char: str = "=") -> None:
    print(char * WIDTH)


def heading(title: str) -> None:
    print()
    line("=")
    print(title.center(WIDTH))
    line("=")


def wrap(text: str, indent: str = "", width: int = 78) -> str:
    value = str(text or "").strip()
    if not value:
        return indent + "(none)"
    return textwrap.fill(
        value,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def bullets(items: Iterable[str], empty: str = "None") -> None:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        print(f"  - {empty}")
        return
    for item in values:
        print(wrap(item, indent="  - "))


def split_concepts(question: dict) -> list[str]:
    return [
        part.strip()
        for part in str(question.get("required_concepts", "")).split("|")
        if part.strip()
    ]


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " ").title())


def is_unanswered(answer: str) -> bool:
    return answer.strip().lower() in SKIP_ANSWERS


def prompt_participant_id() -> str:
    heading("SKILLSCOPE AI - EMPLOYEE ASSESSMENT")
    print("This prototype uses an anonymous participant ID. No name is required.")
    while True:
        try:
            participant_id = input("\nAnonymous participant ID: ").strip()
        except EOFError:
            participant_id = ""
        if participant_id:
            return participant_id
        print("Please enter an anonymous ID, for example DEMO-001.")


def prompt_role() -> str:
    heading("SELECT TECHNICAL COURSE / DOMAIN")
    for index, (_, label) in enumerate(ROLE_OPTIONS, start=1):
        print(f"  {index}. {label}")

    while True:
        try:
            choice = input("\nSelect course [1-5]: ").strip()
        except EOFError:
            choice = ""
        if choice.isdigit() and 1 <= int(choice) <= len(ROLE_OPTIONS):
            return ROLE_OPTIONS[int(choice) - 1][0]
        print("Please enter a number from 1 to 5.")


def collect_answers(assessment_set: dict) -> list[tuple[dict, str]]:
    heading("10-QUESTION ASSESSMENT")
    print(f"Assigned set: {assessment_set['set_id']}")
    print(f"Domain:       {role_label(assessment_set['role'])}")
    print("\nAnswer each question in your own words.")
    print("You may type N/A, I don't know, SKIP, or press ENTER to skip.")
    print("The next question appears after each answer.\n")

    responses: list[tuple[dict, str]] = []
    total = len(assessment_set["questions"])

    for index, question in enumerate(assessment_set["questions"], start=1):
        line("-")
        print(f"QUESTION {index} OF {total}   [{question['question_id']}]")
        print(wrap(question["question"], indent="  "))
        try:
            answer = input("\nYour answer: ").strip()
        except EOFError:
            answer = ""

        if is_unanswered(answer):
            display = answer if answer else "[blank / skipped]"
            print(f"Recorded as unanswered: {display}")
        else:
            print("Answer recorded.")

        responses.append((question, answer))

    return responses


def retrieval_query(question: dict, answer: str) -> str:
    return " ".join(
        [
            question["question"],
            answer,
            question.get("required_concepts", "").replace("|", " "),
        ]
    )


def to_retrieved_rows(retrieved) -> list[dict]:
    return [
        {
            "chunk_id": item.chunk["chunk_id"],
            "retrieval_score": round(item.score, 4),
            "text": item.chunk.get("text", ""),
            "source_title": item.chunk.get("source_title", "Unknown source"),
            "source_url": item.chunk.get("source_url", ""),
        }
        for item in retrieved
    ]


def evaluate_unanswered(question: dict, answer: str, retriever: RoleAwareRetriever, top_k: int) -> QuestionResult:
    """Give unanswered responses zero marks but still retrieve reading evidence."""
    retrieved = retriever.retrieve(
        retrieval_query(question, ""),
        role=question["role"],
        top_k=top_k,
    )
    return QuestionResult(
        question=question,
        answer=answer,
        status="unanswered",
        score=0.0,
        max_score=float(question.get("max_score", 10)),
        strengths=[],
        missing_concepts=split_concepts(question),
        improvement_areas=[
            {
                "area": "Question not answered",
                "priority": "high",
                "reason": "No assessable technical answer was provided.",
                "recommended_activity": "Review the trusted source listed for this question and try the question again.",
            }
        ],
        confidence="high",
        rationale="The response was recorded as N/A, unknown, skipped, or blank.",
        evidence_chunk_ids=[],
        retrieved_evidence=to_retrieved_rows(retrieved),
        mode="unanswered",
    )


def evaluate_answered(
    question: dict,
    answer: str,
    retriever: RoleAwareRetriever,
    client: OllamaClient,
    top_k: int,
) -> QuestionResult:
    """Run the real per-question RAG path and preserve evidence for reporting."""
    retrieved = retriever.retrieve(
        retrieval_query(question, answer),
        role=question["role"],
        top_k=top_k,
    )
    evidence = [item.chunk for item in retrieved]
    if not evidence:
        raise GenerationError("No role-specific evidence was retrieved.")

    prompt = build_augmented_prompt(question, answer, evidence)
    raw = client.generate_json(SYSTEM_PROMPT, prompt)
    validated = validate_generation(
        raw,
        allowed_chunk_ids={row["chunk_id"] for row in evidence},
        max_score=question.get("max_score", 10),
    )

    return QuestionResult(
        question=question,
        answer=answer,
        status="answered",
        score=float(validated["score"]),
        max_score=float(question.get("max_score", 10)),
        strengths=validated.get("strengths", []),
        missing_concepts=validated.get("missing_concepts", []),
        improvement_areas=validated.get("improvement_areas", []),
        confidence=validated.get("confidence", "low"),
        rationale=validated.get("rationale", ""),
        evidence_chunk_ids=validated.get("evidence_chunk_ids", []),
        retrieved_evidence=to_retrieved_rows(retrieved),
        mode="rag",
    )


def evaluate_fallback(
    question: dict,
    answer: str,
    retriever: RoleAwareRetriever,
    top_k: int,
    reason: str,
) -> QuestionResult:
    """Transparent fallback if Ollama is unavailable; clearly labelled non-RAG."""
    retrieved = retriever.retrieve(
        retrieval_query(question, answer),
        role=question["role"],
        top_k=top_k,
    )
    baseline = score_answer(answer, question)
    return QuestionResult(
        question=question,
        answer=answer,
        status="answered",
        score=float(baseline["score"]),
        max_score=float(baseline["max_score"]),
        strengths=[f"Matched concept: {item}" for item in baseline.get("matched_concepts", [])],
        missing_concepts=baseline.get("missing_concepts", []),
        improvement_areas=[
            {
                "area": "Fallback review required",
                "priority": "high",
                "reason": reason,
                "recommended_activity": "Use the trusted reading source below and rerun when Ollama is available.",
            }
        ],
        confidence="low",
        rationale=f"Retrieval-only fallback: {reason}",
        evidence_chunk_ids=[],
        retrieved_evidence=to_retrieved_rows(retrieved),
        mode="retrieval_only_fallback",
    )


def print_rag_progress(index: int, total: int, result: QuestionResult, verbose_rag: bool) -> None:
    question = result.question
    print(f"\n[{index}/{total}] {question['question_id']} - {question['question']}")

    if result.status == "unanswered":
        print("  Response: unanswered -> score 0; evidence retrieved for learning support.")
    else:
        print("  RAG: role filter -> TF-IDF retrieval -> augmented prompt -> Ollama -> validation")

    if result.retrieved_evidence:
        top = result.retrieved_evidence[0]
        print(
            f"  Top evidence: {top['chunk_id']} | {top['source_title']} "
            f"| similarity={top['retrieval_score']:.4f}"
        )
    else:
        print("  Top evidence: none")

    print(f"  Result: {result.score:.1f}/{result.max_score:.0f} | mode={result.mode}")

    if verbose_rag and result.retrieved_evidence:
        print("  Retrieved evidence:")
        for row in result.retrieved_evidence:
            print(
                f"    - {row['chunk_id']} ({row['retrieval_score']:.4f}) "
                f"{row['source_title']}"
            )


def overall_gap(percent: float) -> str:
    if percent >= 75:
        return "none / strong foundation"
    if percent >= 40:
        return "development"
    return "priority"


def reading_candidates(result: QuestionResult) -> list[dict]:
    """Prefer model-cited evidence, otherwise the highest retrieved evidence."""
    if not result.retrieved_evidence:
        return []

    by_id = {row["chunk_id"]: row for row in result.retrieved_evidence}
    selected = [by_id[cid] for cid in result.evidence_chunk_ids if cid in by_id]
    if not selected:
        selected = result.retrieved_evidence[:1]
    return selected


def print_final_report(participant_id: str, role: str, set_id: str, results: list[QuestionResult]) -> None:
    heading("FINAL EVIDENCE-GROUNDED DEVELOPMENT REPORT")

    earned = sum(item.score for item in results)
    possible = sum(item.max_score for item in results)
    percent = (earned / possible * 100.0) if possible else 0.0
    answered = sum(item.status == "answered" for item in results)
    unanswered = len(results) - answered
    rag_count = sum(item.mode == "rag" for item in results)
    fallback_count = sum(item.mode == "retrieval_only_fallback" for item in results)

    print(f"Participant ID: {participant_id}")
    print(f"Course/domain:  {role_label(role)}")
    print(f"Assessment set: {set_id}")
    print(f"Answered:       {answered}/{len(results)}")
    print(f"Skipped/unknown:{unanswered}/{len(results)}")
    print(f"RAG evaluations:{rag_count}")
    if fallback_count:
        print(f"Fallbacks:      {fallback_count} (not full RAG)")

    print("\nOVERALL RESULT")
    print(f"  Score: {earned:.1f}/{possible:.0f} ({percent:.1f}/100)")
    print(f"  Overall learning-gap level: {overall_gap(percent)}")
    print("  Human technical review: REQUIRED")

    print("\nQUESTION-BY-QUESTION")
    for index, result in enumerate(results, start=1):
        marker = "UNANSWERED" if result.status == "unanswered" else result.mode.upper()
        print(
            f"  {index:>2}. {result.question['question_id']}: "
            f"{result.score:.1f}/{result.max_score:.0f}  [{marker}]"
        )

    print("\nKNOWLEDGE GAPS")
    gap_lines: list[str] = []
    seen_gaps: set[str] = set()
    for result in results:
        for area in result.improvement_areas:
            label = str(area.get("area", "")).strip()
            reason = str(area.get("reason", "")).strip()
            key = (label + "|" + reason).lower()
            if label and key not in seen_gaps:
                seen_gaps.add(key)
                gap_lines.append(f"{label}: {reason}" if reason else label)
        for missing in result.missing_concepts:
            missing_text = str(missing).strip()
            key = ("missing|" + missing_text).lower()
            if missing_text and key not in seen_gaps:
                seen_gaps.add(key)
                gap_lines.append(f"Missing concept: {missing_text}")

    bullets(gap_lines[:15], empty="No major gaps identified by the current draft evaluation.")
    if len(gap_lines) > 15:
        print(f"  ... and {len(gap_lines) - 15} additional gap observations.")

    print("\nRECOMMENDED READING")
    reading_rows: list[tuple[str, str, str, str]] = []
    seen_urls: set[str] = set()
    for result in sorted(results, key=lambda row: row.score / row.max_score if row.max_score else 0.0):
        if result.score >= result.max_score:
            continue
        for evidence in reading_candidates(result):
            url = evidence.get("source_url", "").strip()
            title = evidence.get("source_title", "Unknown source").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            reading_rows.append(
                (
                    result.question["question_id"],
                    evidence.get("chunk_id", ""),
                    title,
                    url,
                )
            )

    if not reading_rows:
        print("  - No reading links were available from retrieved evidence.")
    else:
        for index, (question_id, chunk_id, title, url) in enumerate(reading_rows[:10], start=1):
            print(f"  {index}. {title}")
            print(f"     Supports: {question_id} | Evidence: {chunk_id}")
            print(f"     {url}")

    print("\nRAG TRACEABILITY")
    print("  Each answered question was evaluated using role-specific retrieved evidence.")
    print("  The LLM received the question, employee answer, rubric, and retrieved passages.")
    print("  Generated evidence IDs were validated against the chunks actually retrieved.")
    print("  Skipped/unknown responses received zero marks but still received trusted reading evidence.")

    print("\nIMPORTANT")
    print("  This is a developmental draft. A technical director must review scores,")
    print("  evidence, context, and recommendations before any real employee use.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 10-question SkillScope AI RAG assessment demo"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Role-specific evidence chunks retrieved per question (default: {TOP_K_DEFAULT})",
    )
    parser.add_argument(
        "--verbose-rag",
        action="store_true",
        help="Show all retrieved evidence IDs for every question while evaluating.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Stop instead of using the transparent baseline if Ollama generation fails.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    chunks, questions, _ = load_project_data()
    sets = build_assessment_sets(questions, questions_per_set=10)
    retriever = RoleAwareRetriever(chunks)
    client = OllamaClient()

    participant_id = prompt_participant_id()
    role = prompt_role()
    assessment_set = assign_set(participant_id, role, sets)

    print("\nAssignment complete.")
    print(f"  Participant: {participant_id}")
    print(f"  Course:      {role_label(role)}")
    print(f"  Set:         {assessment_set['set_id']} (10 questions)")

    responses = collect_answers(assessment_set)

    heading("RAG EVALUATION")
    print("All 10 responses have been collected.")
    print("Answered questions now run through the real RAG pipeline.")
    print("Skipped/unknown answers receive zero marks and trusted reading evidence.\n")

    results: list[QuestionResult] = []
    total = len(responses)
    ollama_failed = False
    ollama_reason = ""

    for index, (question, answer) in enumerate(responses, start=1):
        if is_unanswered(answer):
            result = evaluate_unanswered(question, answer, retriever, args.top_k)
        elif ollama_failed and not args.no_fallback:
            result = evaluate_fallback(
                question,
                answer,
                retriever,
                args.top_k,
                ollama_reason,
            )
        else:
            try:
                result = evaluate_answered(
                    question,
                    answer,
                    retriever,
                    client,
                    args.top_k,
                )
            except GenerationError as exc:
                if args.no_fallback:
                    raise SystemExit(
                        "\nRAG generation failed. Make sure Ollama is running and the "
                        f"model {client.model} is installed.\nReason: {exc}"
                    )
                ollama_failed = True
                ollama_reason = str(exc)
                print("\n  WARNING: Ollama generation failed.")
                print("  Remaining answered questions will use a clearly labelled retrieval-only fallback.")
                result = evaluate_fallback(
                    question,
                    answer,
                    retriever,
                    args.top_k,
                    ollama_reason,
                )

        results.append(result)
        print_rag_progress(index, total, result, args.verbose_rag)

    print_final_report(
        participant_id=participant_id,
        role=role,
        set_id=assessment_set["set_id"],
        results=results,
    )


if __name__ == "__main__":
    main()
