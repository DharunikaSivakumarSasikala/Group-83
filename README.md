# SkillScope AI v4 — Group 83

**Work Integrated Learning Project | RMIT University | Group 83**

SkillScope AI is a role-aware technical knowledge assessment and learning-support prototype. It identifies knowledge gaps from short role-specific assessments and uses retrieved technical evidence to support focused learning feedback.

The prototype is designed for **learning and development support**, not for automated hiring, promotion, disciplinary, remuneration, or termination decisions.

## Project Overview

SkillScope follows a simple workflow:

**Role-Based Assessment → Concept Check → Knowledge Gap → Evidence Retrieval → Learning Feedback**

The current prototype supports five technical roles:

- Backend Software Engineering
- Frontend/UI Development
- DevOps/SRE
- QA/Testing
- Data Engineering

## Prototype Snapshot

| Component | Current v4 implementation |
|---|---:|
| Technical roles | 5 |
| Knowledge chunks | 120 |
| Written questions | 83 |
| Quiz questions | 25 |
| Evaluation answer cases | 249 |
| Assessment sets | 25 |
| Sets per role | 5 |
| Questions per assessment | 10 |
| Written questions per assessment | 7 |
| Quiz questions per assessment | 3 |
| Automated tests in final suite | 42 |

## Main Features

- Signed role-specific candidate invitation links
- Five deterministic assessment sets per role
- Mixed assessment format: 7 written questions + 3 quiz questions
- Semantic written-answer checking against predefined concepts
- Concept labels: `demonstrated`, `absent`, and `contradicted`
- Python-controlled numeric score calculation
- Deterministic concept-matching fallback
- Role-aware TF-IDF retrieval baseline
- Role-aware semantic retrieval using `nomic-embed-text`
- Top-5 evidence retrieval for downstream feedback
- Local RAG feedback using `llama3.2:3b`
- Citation-ID validation and safe fallback behaviour
- Detailed question-by-question results
- Human-review-required output
- Retrieval and RAG evaluation utilities
- Automated testing for assessment, retrieval, scoring, RAG, invitations, and integrity

## How the Scoring Works

Written questions contain predefined technical concepts.

The local `llama3.2:3b` model is used as a constrained semantic validation layer. For each required concept, it returns one of:

- **demonstrated** — the employee expresses the correct technical meaning
- **absent** — the concept is not sufficiently demonstrated
- **contradicted** — the answer makes a conflicting or technically incorrect claim

The language model does **not** freely invent a numeric score. Python calculates the percentage from the validated concept labels.

A written answer is treated as competent when it reaches the prototype threshold of **80%** and contains no contradicted required concept.

If semantic validation is unavailable, the application falls back to the deterministic concept matcher and keeps the result marked for human review.

Quiz questions use exact scoring:

- correct answer: 100%
- incorrect or skipped answer: 0%

## Retrieval and RAG

### TF-IDF baseline

TF-IDF is retained as a lightweight lexical baseline and fallback.

### Semantic retrieval

The preferred v4 retriever uses the local `nomic-embed-text` model.

The knowledge base is first filtered by technical role and then ranked by semantic similarity. The top five chunks are returned as evidence.

### RAG learning feedback

For selected weak written responses, SkillScope retrieves role-specific evidence and provides it to `llama3.2:3b` together with:

- the assessment question
- employee answer
- required concepts
- retrieved evidence

The generated feedback is developmental. Citation identifiers are checked against the evidence actually retrieved.

RAG is used to support **what the employee should learn next**; the assessment component identifies **what the employee currently demonstrates**.

## Development Evaluation

A 60-case low-keyword-overlap semantic retrieval stress test was used during development.

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Miss@5 |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 21.67% | 35.00% | 48.33% | 0.3058 | 51.67% |
| Embedding | 33.33% | 75.00% | 86.67% | 0.5444 | 13.33% |

Top-five misses fell from **31 to 8**.

These are **development results**, not final independently human-reviewed real-world accuracy.

The final automated verification suite currently passes **42/42 tests**. This verifies defined software behaviour; it does not mean that the AI components are 100% accurate.

## Team Members and Functional Responsibilities

The responsibilities below describe the team's functional areas and how they align with the current v4 codebase. They should not be interpreted as exact file authorship unless confirmed by Git commit history.

| Team member | Main responsibility | Related implementation areas |
|---|---|---|
| **Sarath Pon Kumar Murugesan** | Project coordination, RAG integration, stakeholder requirements, evaluation design | RAG pipeline, RAG-quality evaluation, grounded-feedback workflow |
| **Santhiya Ravi** | Knowledge-base curation, preprocessing/metadata, source quality, ethics and privacy | Knowledge-base records, source metadata/provenance, safety and evaluation documentation |
| **Satyajit Seetharaman Kavitha** | Embeddings, retrieval, ranking and retrieval evaluation | TF-IDF/embedding retrievers, retrieval evaluation, semantic/hybrid retrieval experiments |
| **Yaswanth Gunta** | Web interface, candidate assessment workflow and usability | Candidate web application, invitation/demo flow, assessment interaction |
| **Dharunika Sivakumar Sasikala** | Question bank and rubrics, quiz scoring and detailed results, QA/testing and documentation | Assessment questions/quizzes, assessment-set generation, detailed results, automated testing and documentation |

## Project Structure

```text
Group-83/
├── README.md
├── requirements.txt
├── skillscope_core.py
├── semantic_scoring.py
├── rag_pipeline.py
├── web_app.py
├── run_demo.py
├── create_invitation.py
├── build_question_sets.py
├── evaluate_retrieval.py
├── hard_semantic_retrieval_test.py
├── larger_semantic_retrieval_test.py
├── quick_embedding_hybrid_test.py
├── evaluate_rag_quality.py
├── build_human_review_set.py
├── clean_human_review_set.py
├── prepare_reviewer_packets.py
├── merge_reviews_for_adjudication.py
├── final_single_reviewer_evaluation.py
├── test_prototype.py
├── test_rag_pipeline.py
├── test_v3_features.py
├── test_embedding_retriever.py
├── test_concept_matching.py
├── test_assessment_integrity.py
├── test_semantic_scoring.py
├── data/
│   ├── assessment_sets.json
│   ├── rag_quality_cases.example.jsonl
│   └── source/
│       ├── knowledge_base.jsonl
│       ├── assessment_questions.jsonl
│       ├── coding_quizzes.jsonl
│       └── evaluation_cases.jsonl
├── results/
├── RAG_ARCHITECTURE.md
├── HALLUCINATION_EVALUATION.md
├── MENTOR_UPDATE.md
└── RELEASE_NOTES_V3.md
```

Some historical scripts and documentation retain earlier v3 names because v4 was developed incrementally from the v3 prototype.

## Requirements

### Python

Python 3.11+ is recommended.

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Current Python requirements:

- NumPy
- scikit-learn

### Ollama

Ollama must be installed separately for semantic retrieval/scoring and RAG generation.

Pull the required local models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Start Ollama:

```bash
ollama serve
```

## Run the Prototype

The easiest option is:

```bash
python run_demo.py
```

By default, this:

1. starts the SkillScope web application;
2. uses embedding retrieval;
3. creates a signed assessment invitation;
4. opens the assessment in the browser.

You can choose a role:

```bash
python run_demo.py --participant DEMO-001 --role data_engineer
```

Available role codes:

```text
backend_software_engineer
frontend_ui_developer
devops_sre_engineer
qa_test_engineer
data_engineer
```

To use the TF-IDF baseline:

```bash
python run_demo.py --retriever tfidf
```

## Run the Application Manually

Start the application:

```bash
python web_app.py
```

In another terminal, create an invitation:

```bash
python create_invitation.py --participant EMP-001 --role qa_test_engineer
```

Open the signed URL printed in the terminal.

## Run the Automated Tests

```bash
python -m unittest -v \
test_prototype.py \
test_rag_pipeline.py \
test_v3_features.py \
test_embedding_retriever.py \
test_concept_matching.py \
test_assessment_integrity.py \
test_semantic_scoring.py
```

Final v4 verification:

```text
Ran 42 tests
OK
```

## Important Limitations

- The current knowledge base contains manually curated chunks rather than an automated raw-document ingestion pipeline.
- The 80% competency threshold is a prototype design choice, not an industry-validated standard.
- Semantic concept labels can still be incorrect.
- RAG and citation validation reduce risk but do not guarantee that generated feedback is fully grounded.
- Development retrieval tests are not a final independent human benchmark.
- Assessment attempts and results are currently held in application memory rather than a production database.
- Production authentication, persistent storage, security controls, larger-scale testing, and independent human evaluation are still required.
- SkillScope should be used as learning support rather than as an automatic employment decision system.

## Main Documentation

- `RAG_ARCHITECTURE.md` — RAG pipeline and evidence flow
- `HALLUCINATION_EVALUATION.md` — proposed grounding/hallucination evaluation
- `MENTOR_UPDATE.md` — project progress summary
- `results/` — development evaluation outputs

## Repository

Project repository:

https://github.com/DharunikaSivakumarSasikala/Group-83

## Academic Use

This repository contains the Group 83 Work Integrated Learning project prototype. Development evaluation results should be interpreted together with the limitations documented in the written report.

---

**Group 83 — SkillScope AI**
