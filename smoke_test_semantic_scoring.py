"""Live Ollama smoke test for SkillScope semantic written-answer validation."""

from skillscope_core import load_project_data
from semantic_scoring import SemanticAnswerValidator


CHUNKS, QUESTIONS, _ = load_project_data()
CHUNK_BY_ID = {row["chunk_id"]: row for row in CHUNKS}
QUESTION_BY_ID = {row["question_id"]: row for row in QUESTIONS}

CASES = [
    (
        QUESTION_BY_ID["Q_DATA_11"],
        "The ingestion pipeline should verify content identifiers, how the metadata is organised, when the content is available, and its language information.",
        "correct paraphrase",
    ),
    (
        QUESTION_BY_ID["Q_DATA_10"],
        "Pipeline metrics should avoid unbounded labels because high cardinality makes the monitoring system efficient.",
        "keyword contradiction",
    ),
    (
        QUESTION_BY_ID["Q_DATA_12"],
        "Use stable identifiers for each series and episode so the same content is not created more than once.",
        "partial answer",
    ),
]

validator = SemanticAnswerValidator(CHUNK_BY_ID)
results = validator.validate_batch([(q, answer) for q, answer, _ in CASES])

print("=" * 72)
print("SKILLSCOPE SEMANTIC SCORING SMOKE TEST")
print("=" * 72)
for question, answer, label in CASES:
    result = results[question["question_id"]]
    print(f"\nCase       : {label}")
    print(f"Question   : {question['question']}")
    print(f"Answer     : {answer}")
    print(f"Accuracy   : {result['accuracy']}%")
    print(f"Competent  : {result['competent']}")
    print(f"Matched    : {result['matched_concepts']}")
    print(f"Missing    : {result['missing_concepts']}")
    print(f"Contradict : {result['contradicted_concepts']}")
    print(f"Consistency: {result['overall_consistency']}")
    print(f"Confidence : {result['semantic_confidence']}")

print("\nExpected behaviour:")
print("- correct paraphrase: 100%")
print("- keyword contradiction: below 80% and not competent")
print("- partial answer: below 80% and not competent")
