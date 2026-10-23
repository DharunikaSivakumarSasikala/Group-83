# SkillScope AI Hallucination Control and Evaluation

## Short faculty answer

We cannot honestly guarantee that a language model never hallucinates. Instead, SkillScope limits what the model is allowed to do, validates its citations, measures unsupported claims on human-labelled cases, and keeps the final accuracy score deterministic. Ollama explains learning gaps; it does not independently decide the employee's final percentage.

## How the risk is reduced

1. Role filtering restricts retrieval to the candidate's assigned technical domain.
2. The model receives only the question, employee answer, required concepts and top retrieved approved passages.
3. Employee text is labelled untrusted so instructions inside an answer must not be followed.
4. Temperature is zero to reduce random variation.
5. The response must follow a fixed JSON schema.
6. Citation IDs are accepted only when they belong to the retrieved evidence set.
7. Scores are bounded between zero and the question maximum.
8. Expected answers and learning URLs come from curated data, not model invention.
9. If Ollama is offline or produces invalid output, the result is labelled `retrieval_only_fallback` rather than being presented as RAG.
10. Every report is marked `human_review_required`.

## Evaluation layers

### Retrieval quality

- Hit@1, Hit@3 and Hit@5 measure whether the expected evidence appears in the first results.
- MRR@5 rewards placing the first relevant passage near the top.
- Miss or unanswered rate measures how often expected evidence is absent from the top five.

Retrieval must be measured separately because generation cannot be grounded when the correct passage was never retrieved.

### Generation grounding

- Citation precision: valid retrieved citations divided by all generated citations.
- Citation recall: expected evidence citations found divided by expected citations.
- Claim groundedness: model claims supported by cited evidence divided by all checked claims.
- Hallucination rate: unsupported checked claims divided by all checked claims, equivalent to `1 - claim groundedness`.

Claim support requires a human reviewer or a separately validated judge. Citation presence alone does not prove that a sentence is supported.

### Scoring quality

- Mean absolute error compares model score with technical-reviewer score.
- Competency agreement measures whether model and reviewer agree on the 80% threshold.
- Weighted Cohen's kappa can be added when reviewers use ordered performance bands.

### Reliability

- JSON-valid output rate.
- Citation-validation failure rate.
- Fallback or abstention rate.
- Repeated-run consistency using the same question, answer and evidence.

## Required experiment

At least two technical reviewers should label a held-out set of answers. They should record the human score, whether each generated claim is supported, which citations are relevant and whether the learning recommendation follows from the evidence. The team should report the metrics produced by `evaluate_rag_quality.py` and disagreements between reviewers. The included example JSONL demonstrates the calculation only and must not be reported as final model performance.
