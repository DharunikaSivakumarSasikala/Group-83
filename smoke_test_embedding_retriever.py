#!/usr/bin/env python3
"""Quick local smoke test for the configured SkillScope retriever."""

from skillscope_core import build_retriever, load_project_data, reference_chunk_ids


def main() -> None:
    chunks, questions, _ = load_project_data()
    retriever = build_retriever(chunks)

    question = questions[0]
    query = " ".join(
        [
            question["question"],
            question.get("required_concepts", "").replace("|", " "),
        ]
    )

    results = retriever.retrieve(
        query,
        role=question["role"],
        top_k=5,
    )

    print("=" * 72)
    print("SKILLSCOPE RETRIEVER SMOKE TEST")
    print("=" * 72)
    print(f"Retriever : {getattr(retriever, 'name', retriever.__class__.__name__)}")
    if getattr(retriever, "name", "") == "embedding":
        print(f"Model     : {retriever.embedder.model_name}")
        print(
            "Cache     : "
            + ("loaded existing vectors" if retriever.cache_loaded else "created new vectors")
        )
        print(f"Cache file: {retriever.cache_path}")

    print(f"Role      : {question['role']}")
    print(f"Question  : {question['question']}")
    expected_ids = reference_chunk_ids(question)
    print(f"Expected  : {', '.join(expected_ids)}")
    print("\nTop 5:")
    for index, item in enumerate(results, start=1):
        marker = "  <-- expected" if item.chunk["chunk_id"] in expected_ids else ""
        print(
            f"{index}. {item.chunk['chunk_id']}  score={item.score:.4f}"
            f"  skill={item.chunk.get('skill', '')}{marker}"
        )


if __name__ == "__main__":
    main()
