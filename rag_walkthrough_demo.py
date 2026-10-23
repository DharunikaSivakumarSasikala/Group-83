

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Iterable

from rag_pipeline import (
    GenerationError,
    OllamaClient,
    SYSTEM_PROMPT,
    build_augmented_prompt,
    validate_generation,
)
from skillscope_core import RoleAwareRetriever, load_project_data


ROLE_LABELS = {
    "backend_software_engineer": "Backend Software Engineering",
    "frontend_ui_developer": "Frontend / UI Development",
    "devops_sre_engineer": "DevOps / SRE",
    "qa_test_engineer": "QA / Testing",
    "data_engineer": "Data Engineering",
}

WIDTH = 76


def line(char: str = "=") -> None:
    print(char * WIDTH)


def heading(title: str) -> None:
    print()
    line("=")
    print(title.center(WIDTH))
    line("=")


def pause(enabled: bool, message: str = "Press ENTER to continue...") -> None:
    if enabled:
        try:
            input(f"\n{message}")
        except EOFError:
            pass


def wrap(text: str, indent: str = "", width: int = 72) -> str:
    text = str(text or "").strip()
    if not text:
        return indent + "(none)"
    return textwrap.fill(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def bullets(items: Iterable[str], empty_text: str = "None") -> None:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        print(f"  - {empty_text}")
        return
    for item in values:
        print(wrap(item, indent="  - "))


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " ").title())


def split_concepts(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def list_questions(questions: list[dict]) -> None:
    heading("AVAILABLE ASSESSMENT QUESTIONS")
    grouped: dict[str, list[dict]] = {}
    for question in questions:
        grouped.setdefault(question["role"], []).append(question)

    for role in sorted(grouped):
        print(f"\n{role_label(role)}")
        print("-" * len(role_label(role)))
        for question in sorted(grouped[role], key=lambda row: row["question_id"]):
            print(f"{question['question_id']}: {question['question']}")


def build_retrieval_query(question: dict, employee_answer: str) -> str:
    """Mirror the query construction used by RAGEvaluator.evaluate()."""
    return " ".join(
        [
            question["question"],
            employee_answer,
            question.get("required_concepts", "").replace("|", " "),
        ]
    )


def print_intro(question: dict, employee_answer: str) -> None:
    heading("SKILLSCOPE AI - RAG WALKTHROUGH")
    print("\nEMPLOYEE CONTEXT")
    print(f"  Role:        {role_label(question['role'])}")
    print(f"  Question ID: {question['question_id']}")
    print(f"  Max score:   {question.get('max_score', 10)}")

    print("\nASSESSMENT QUESTION")
    print(wrap(question["question"], indent="  "))

    print("\nEMPLOYEE ANSWER")
    if employee_answer.strip():
        print(wrap(employee_answer, indent="  "))
    else:
        print("  [No answer provided]")

    print("\nRUBRIC CONCEPTS")
    bullets(split_concepts(question.get("required_concepts", "")))


def print_retrieval(retrieved) -> None:
    heading("STEP 1 - RETRIEVAL")
    print("The system first searches only the approved knowledge chunks for the")
    print("employee's technical role. TF-IDF + cosine similarity ranks the chunks.")
    print("The top results become the evidence supplied to the LLM.\n")

    for rank, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        print(f"[{rank}] {chunk['chunk_id']}  |  similarity={item.score:.4f}")
        print(f"    Source: {chunk.get('source_title', 'Unknown source')}")
        print(wrap(chunk.get("text", ""), indent="    ", width=72))
        print(f"    URL: {chunk.get('source_url', '')}")
        if rank != len(retrieved):
            print()


def print_augmentation(question: dict, employee_answer: str, evidence: list[dict], prompt: str, show_full_prompt: bool) -> None:
    heading("STEP 2 - AUGMENTATION")
    print("PROMPT COMPONENTS")
    print(f"  Role: {role_label(question['role'])}")
    print(f"  Question: {question['question']}")
    print(f"  Employee answer: {employee_answer if employee_answer.strip() else '[blank]'}")
    print(f"  Maximum score: {question.get('max_score', 10)}")
    print(f"  Required concepts: {', '.join(split_concepts(question.get('required_concepts', '')))}")
    print("  Retrieved evidence IDs: " + ", ".join(row["chunk_id"] for row in evidence))

    if show_full_prompt:
        print("\nFULL AUGMENTED USER PROMPT")
        line("-")
        print(prompt)
        line("-")
    else:
        print("\nFor the mentor demo, the full prompt is hidden for readability.")
        print("Use --show-full-prompt if you want to display it.")


def print_generation_start(client: OllamaClient) -> None:
    heading("STEP 3 - GENERATION")
    print("The augmented prompt is now sent to the local Ollama model.")
    print(f"  Model:       {client.model}")
    print(f"  Endpoint:    {client.endpoint}")
    print("  Temperature: 0")
    print("  Output:      structured JSON")
    print("\nGenerating evidence-grounded evaluation...")


def print_validation(raw: dict, validated: dict, allowed_ids: set[str]) -> None:
    heading("STEP 4 - VALIDATION")
    cited = validated.get("evidence_chunk_ids", [])
    invalid = sorted(set(cited) - allowed_ids)
    
    print("It checks the score, confidence, JSON structure, and evidence citations.\n")
    print(f"  Generated score:       {raw.get('score', '(missing)')}")
    print(f"  Validated score:       {validated.get('score')}")
    print(f"  Model evidence IDs:    {', '.join(cited) if cited else '(none)'}")
    print(f"  Invalid evidence IDs:  {', '.join(invalid) if invalid else 'None'}")
    print(f"  Model review flag:     {validated.get('needs_human_review')}")
    print("  Application policy:   HUMAN REVIEW REQUIRED")
    print("\nValidation passed: all accepted evidence IDs came from retrieval.")


def print_result(question: dict, validated: dict, retrieved) -> None:
    heading("STEP 5 - EVIDENCE-GROUNDED EMPLOYEE RESULT")

    max_score = question.get("max_score", 10)
    print(f"DRAFT SCORE: {validated['score']}/{max_score}")
    print(f"CONFIDENCE:  {validated['confidence'].upper()}")
    print("MODE:        RAG")
    print("HUMAN REVIEW: REQUIRED")

    print("\nSTRENGTHS")
    bullets(validated.get("strengths", []))

    print("\nMISSING CONCEPTS")
    bullets(validated.get("missing_concepts", []))

    print("\nIMPROVEMENT AREAS AND LEARNING ACTIVITIES")
    areas = validated.get("improvement_areas", [])
    if not areas:
        print("  - None generated")
    else:
        for index, area in enumerate(areas, start=1):
            print(f"  {index}. {area.get('area') or 'Improvement area'}")
            print(f"     Priority: {area.get('priority', 'medium')}")
            print(wrap(area.get("reason", ""), indent="     Reason: ", width=72))
            print(wrap(area.get("recommended_activity", ""), indent="     Activity: ", width=72))

    print("\nRATIONALE")
    print(wrap(validated.get("rationale", ""), indent="  "))

    print("\nEVIDENCE USED BY THE MODEL")
    cited_ids = set(validated.get("evidence_chunk_ids", []))
    retrieved_by_id = {item.chunk["chunk_id"]: item for item in retrieved}
    if not cited_ids:
        print("  - The model did not cite a retrieved chunk.")
    else:
        for chunk_id in validated.get("evidence_chunk_ids", []):
            item = retrieved_by_id[chunk_id]
            chunk = item.chunk
            print(f"  - {chunk_id} | {chunk.get('source_title', 'Unknown source')}")
            print(f"    Retrieval similarity: {item.score:.4f}")
            print(wrap(chunk.get("text", ""), indent="    ", width=72))
            print(f"    Source: {chunk.get('source_url', '')}")


def print_trace(validated: dict, retrieved) -> None:
    heading("STEP 6 - RAG TRACEABILITY")
    retrieved_ids = [item.chunk["chunk_id"] for item in retrieved]
    cited_ids = validated.get("evidence_chunk_ids", [])

    print("RETRIEVED")
    print("  " + " -> ".join(retrieved_ids))
    print("\nCITED IN GENERATED EVALUATION")
    print("  " + (", ".join(cited_ids) if cited_ids else "No evidence IDs cited"))

    print("\nTRACE CHECK")
    if all(chunk_id in retrieved_ids for chunk_id in cited_ids):
        print("  PASS - every cited chunk was actually retrieved.")
    else:
        print("  FAIL - an unretrieved chunk was cited.")

    print("\nWHAT THIS DEMONSTRATES")
    print("  Employee answer")
    print("       -> role-aware retrieval")
    print("       -> trusted evidence")
    print("       -> augmented prompt")
    print("       -> local Ollama generation")
    print("       -> validation")
    print("       -> strengths, gaps and learning recommendation")
    print("       -> mandatory human review")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive SkillScope AI RAG walkthrough for mentor demonstrations"
    )
    parser.add_argument(
        "--question-id",
        default="Q_QA_T_01",
        help="Assessment question ID (default: Q_QA_T_01)",
    )
    parser.add_argument(
        "--answer",
        help="Employee answer. If omitted, the demo asks for it interactively.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of role-specific evidence chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run all stages without waiting for ENTER between them.",
    )
    parser.add_argument(
        "--show-full-prompt",
        action="store_true",
        help="Display the complete augmented user prompt before generation.",
    )
    parser.add_argument(
        "--list-questions",
        action="store_true",
        help="List all available assessment questions and exit.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    chunks, questions, _ = load_project_data()

    if args.list_questions:
        list_questions(questions)
        return

    question_by_id = {row["question_id"]: row for row in questions}
    if args.question_id not in question_by_id:
        print(f"Unknown question ID: {args.question_id}\n", file=sys.stderr)
        print("Use --list-questions to see valid IDs.", file=sys.stderr)
        raise SystemExit(2)

    question = question_by_id[args.question_id]

    if args.answer is None:
        heading("SKILLSCOPE AI - EMPLOYEE INPUT")
        print(f"Role: {role_label(question['role'])}")
        print(f"Question: {question['question']}")
        try:
            employee_answer = input("\nEmployee answer: ").strip()
        except EOFError:
            employee_answer = ""
    else:
        employee_answer = args.answer.strip()

    pauses = not args.no_pause
    print_intro(question, employee_answer)
    pause(pauses, "Press ENTER to begin RAG retrieval...")

    retriever = RoleAwareRetriever(chunks)
    query = build_retrieval_query(question, employee_answer)
    retrieved = retriever.retrieve(query, role=question["role"], top_k=args.top_k)
    if not retrieved:
        raise SystemExit("No role-specific evidence was retrieved.")

    print_retrieval(retrieved)
    pause(pauses, "Press ENTER to build the augmented prompt...")

    evidence = [item.chunk for item in retrieved]
    prompt = build_augmented_prompt(question, employee_answer, evidence)
    print_augmentation(question, employee_answer, evidence, prompt, args.show_full_prompt)
    pause(pauses, "Press ENTER to send the augmented prompt to Ollama...")

    client = OllamaClient()
    print_generation_start(client)

    try:
        raw = client.generate_json(SYSTEM_PROMPT, prompt)
        validated = validate_generation(
            raw,
            allowed_chunk_ids={row["chunk_id"] for row in evidence},
            max_score=question.get("max_score", 10),
        )
    except GenerationError as exc:
        print("\nRAG GENERATION FAILED")
        print(wrap(str(exc), indent="  "))
        print("\nCheck that Ollama is running and that llama3.2:3b is installed:")
        print("  ollama list")
        print("  ollama pull llama3.2:3b")
        print("  ollama serve")
        raise SystemExit(1)

    print("Generation complete.")
    pause(pauses, "Press ENTER to validate the generated result...")

    print_validation(raw, validated, {row["chunk_id"] for row in evidence})
    pause(pauses, "Press ENTER to view the employee development result...")

    print_result(question, validated, retrieved)
    pause(pauses, "Press ENTER to view RAG evidence traceability...")

    print_trace(validated, retrieved)


if __name__ == "__main__":
    main()
