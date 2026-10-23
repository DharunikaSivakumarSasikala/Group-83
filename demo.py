"""Interactive command-line proof of concept for a mentor demonstration."""

from skillscope_core import (
    build_retriever,
    assign_set,
    build_assessment_sets,
    load_project_data,
    score_answer,
)


ROLE_NAMES = {
    "1": "backend_software_engineer",
    "2": "frontend_ui_developer",
    "3": "devops_sre_engineer",
    "4": "qa_test_engineer",
    "5": "data_engineer",
}


def main() -> None:
    chunks, questions, _ = load_project_data()
    retriever = build_retriever(chunks)
    sets = build_assessment_sets(questions)
    print("\nSkillScope AI — employee assessment demo")
    print(f"Retriever: {getattr(retriever, 'name', retriever.__class__.__name__)}")
    print("1 Backend  2 Frontend/UI  3 DevOps/SRE  4 QA/Testing  5 Data Engineering")
    role = ROLE_NAMES.get(input("Choose role [1-5]: ").strip(), ROLE_NAMES["4"])
    participant = input("Anonymous participant ID: ").strip() or "DEMO-001"
    assigned = assign_set(participant, role, sets)
    question = assigned["questions"][0]
    print(f"\nAssigned set: {assigned['set_id']}")
    print(f"Question: {question['question']}")
    answer = input("Your answer: ").strip()
    scored = score_answer(answer, question)
    evidence = retriever.retrieve(question["question"], role=role, top_k=3)
    print(f"\nDraft score: {scored['score']}/{scored['max_score']} ({scored['gap']} gap)")
    print("Matched concepts:", ", ".join(scored["matched_concepts"]) or "none")
    print("Missing concepts:", ", ".join(scored["missing_concepts"]) or "none")
    print("\nRetrieved evidence:")
    for item in evidence:
        print(f"- {item.chunk['chunk_id']} ({item.score:.3f}): {item.chunk['text']}")
        print(f"  Source: {item.chunk['source_url']}")
    print("\nThis is a draft decision-support result and requires technical-director review.")


if __name__ == "__main__":
    main()
