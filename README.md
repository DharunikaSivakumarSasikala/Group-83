# SkillScope AI v4 — Group 83

**Work Integrated Learning Project | RMIT University | Group 83**

SkillScope AI is a role-aware technical knowledge assessment and learning-support prototype. It identifies knowledge gaps from short role-specific assessments and uses retrieved technical evidence to support focused learning feedback.

The prototype is designed for **learning and development support**. It is not intended to make automated decisions about hiring, promotion, remuneration, discipline, or termination.

---

## Project Overview

Employees working in the same technical role do not always have the same knowledge gaps. One employee may already understand APIs but need support with reliability, while another may require help with data quality or testing.

SkillScope AI was developed to help identify these differences and provide more focused learning support.

The overall workflow is:

**Role-Based Assessment → Concept Check → Knowledge Gap → Evidence Retrieval → Learning Feedback**

The current prototype supports five technical roles:

- Backend Software Engineering
- Frontend/UI Development
- DevOps/SRE
- QA/Testing
- Data Engineering

---

## Project Objectives

SkillScope AI aims to:

- provide short role-specific technical assessments;
- identify technical concepts that employees demonstrate, miss, or contradict;
- recognise correct answers even when different wording is used;
- retrieve technical evidence relevant to an identified knowledge gap;
- provide focused learning feedback based on retrieved evidence;
- compare lexical and semantic retrieval approaches;
- keep scoring and feedback processes understandable and reviewable;
- support human decision-making rather than replace technical reviewers.

---

## Prototype Summary

| Component | Current v4 Implementation |
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
| Final automated tests | 42 |

---

# Main Features

SkillScope AI v4 includes:

- role-specific employee assessments;
- five assessment sets for each technical role;
- deterministic assessment-set assignment;
- 7 written questions and 3 quiz questions per assessment;
- predefined concepts for written questions;
- semantic written-answer evaluation;
- concept labels:
  - `demonstrated`
  - `absent`
  - `contradicted`
- Python-controlled score calculation;
- deterministic concept-matching fallback;
- exact scoring for quiz questions;
- role-aware TF-IDF retrieval baseline;
- role-aware semantic embedding retrieval;
- local `nomic-embed-text` embeddings;
- local `llama3.2:3b` semantic validation;
- Retrieval-Augmented Generation (RAG);
- evidence-supported learning recommendations;
- top-5 evidence retrieval;
- citation validation;
- structured RAG output validation;
- safe fallback behaviour;
- detailed employee results;
- signed assessment invitations;
- assessment-integrity checks;
- retrieval evaluation tools;
- semantic retrieval stress testing;
- RAG-quality evaluation tools;
- human-review dataset preparation;
- automated testing across the main prototype components.

---

# How SkillScope Works

## 1. Role-Based Assessment

An employee is assigned one of the five supported technical roles and completes a ten-question assessment.

Each assessment contains:

- **7 written-response questions**
- **3 quiz questions**

The written questions allow employees to explain technical knowledge in their own words.

Quiz questions provide a more controlled check of specific technical knowledge.

---

## 2. Required Concepts

Each written question contains predefined technical concepts that represent the important knowledge expected in the answer.

For example:

> Which metadata should an OTT ingestion pipeline validate?

The expected concepts include:

- identifiers;
- hierarchy;
- availability;
- language.

The employee does not need to use these exact words.

SkillScope is designed to check whether the same technical meaning is demonstrated.

---

# Written-Answer Scoring

Earlier versions of the prototype relied more heavily on direct concept and keyword matching.

Testing identified two important problems:

1. A technically correct answer could be missed when the employee used different wording.
2. An incorrect answer could contain the expected keywords and still receive credit.

For example:

> High cardinality makes monitoring more efficient.

This answer contains the expected term `high cardinality`, but the technical meaning is incorrect.

To improve this behaviour, SkillScope v4 uses a constrained semantic evaluation approach.

The local `llama3.2:3b` model checks only the predefined concepts associated with the question.

Each concept is classified as:

### Demonstrated

The employee correctly explains or shows understanding of the required concept.

### Absent

The concept is not clearly demonstrated.

### Contradicted

The employee provides an explanation that conflicts with the required technical idea.

---

## Numeric Score Calculation

The language model does not freely assign the final numeric score.

Python calculates the score from the concept labels.

If:

- `d` = number of demonstrated concepts
- `n` = total required concepts

then:

```text
Accuracy = (d / n) × 100
```

A written response is considered competent when:

- the score is at least **80%**; and
- there are no contradicted required concepts.

The 80% threshold is a prototype design decision and is not presented as an industry-validated competency standard.

---

## Fallback Scoring

If semantic evaluation is unavailable or returns an invalid response, SkillScope uses the deterministic concept-matching approach as a fallback.

The fallback supports:

- basic concept matching;
- controlled aliases;
- simple word variations;
- plural forms;
- limited typo tolerance.

Fallback results remain suitable for human review.

---

# Quiz Scoring

Quiz questions are evaluated using their predefined correct option.

- Correct answer: **100%**
- Incorrect answer: **0%**
- Skipped answer: **0%**

Quiz questions do not use semantic scoring.

---

# Knowledge Base

The SkillScope knowledge base contains **120 role-tagged technical knowledge chunks**.

The supported roles are:

- Backend Software Engineering
- Frontend/UI Development
- DevOps/SRE
- QA/Testing
- Data Engineering

Technical material was selected mainly from official, standards-based, or maintainer documentation.

Examples include:

- OpenAPI
- OWASP
- MDN
- W3C
- Docker
- Kubernetes
- Playwright
- pytest
- Apache Airflow
- Apache Kafka
- Prometheus
- Apple technical documentation

Each knowledge chunk contains information such as:

- role;
- skill;
- proficiency level;
- technical content;
- source organisation;
- source title;
- source URL;
- access information.

---

# Retrieval

SkillScope uses role-aware retrieval.

Before ranking knowledge chunks, the system filters the knowledge base according to the employee's assigned technical role.

This reduces the chance of returning technically related but role-inappropriate information.

Two retrieval approaches were investigated.

---

## TF-IDF Baseline

TF-IDF is retained as the lexical retrieval baseline.

It ranks knowledge chunks mainly according to the important words shared between:

- the query; and
- the stored technical information.

TF-IDF is:

- lightweight;
- reproducible;
- fast;
- useful for comparison.

However, it can struggle when the query and the relevant knowledge source use different words to describe the same technical idea.

---

## Semantic Embedding Retrieval

The preferred v4 retrieval method uses:

```text
nomic-embed-text
```

through a local Ollama environment.

The model converts:

- the retrieval query; and
- knowledge chunks

into numerical representations of their meaning.

The system then compares these representations to identify semantically similar content.

This makes retrieval less dependent on exact keyword overlap.

The search is first restricted by role and then ranked by semantic similarity.

The top **5** results are returned for downstream use.

---

# Retrieval Evaluation

A 60-case low-keyword-overlap semantic retrieval stress test was used during development.

The test included:

- 60 total cases;
- 5 technical roles;
- 12 cases per role;
- low lexical overlap between the query and expected source.

The same cases were evaluated using TF-IDF and embedding retrieval.

## Results

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Miss@5 |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 21.67% | 35.00% | 48.33% | 0.3058 | 51.67% |
| Embedding | 33.33% | 75.00% | 86.67% | 0.5444 | 13.33% |

Top-five misses:

```text
TF-IDF:     31
Embedding:   8
```

This represents approximately a **74% reduction in top-five misses**.

These results support the use of semantic embedding retrieval for difficult paraphrased queries in the development test.

They are **development results** and should not be interpreted as final independently human-reviewed real-world accuracy.

---

# Retrieval-Augmented Generation

SkillScope uses Retrieval-Augmented Generation (RAG) to support focused learning feedback.

The workflow is:

```text
Assessment Question
        +
Employee Answer
        +
Required Concepts
        ↓
Role-Aware Retrieval
        ↓
Top Relevant Evidence
        ↓
llama3.2:3b
        ↓
Learning Feedback
```

The RAG component is mainly used to answer:

> What should the employee learn next?

The assessment component answers:

> What does the employee currently understand?

These two functions are kept separate.

---

# RAG Safety and Validation

Several controls are used in the RAG pipeline.

These include:

- role filtering;
- bounded retrieved evidence;
- structured model output;
- citation identifier validation;
- score-bound validation;
- fallback behaviour;
- human-review-required results.

The generated citation identifiers must refer to knowledge chunks that were actually retrieved.

If a valid generated response cannot be produced, SkillScope can return a fallback result instead.

These controls reduce risk, but they do not guarantee that all generated feedback is correct.

Independent human evaluation is still required.

---

# Local Models

SkillScope v4 uses two main local models.

## `nomic-embed-text`

Used for:

- semantic retrieval;
- knowledge-chunk embeddings;
- query embeddings;
- similarity ranking.

## `llama3.2:3b`

Used for:

- constrained semantic concept checking;
- written-answer interpretation;
- RAG learning feedback;
- structured generation.

Python remains responsible for:

- numeric score calculation;
- competency rules;
- quiz scoring;
- fallback behaviour;
- output validation.

---

# Assessment Assignment

SkillScope contains **25 assessment sets**:

```text
5 roles × 5 sets per role = 25 sets
```

Each set contains:

```text
7 written questions
+
3 quiz questions
=
10 questions
```

The participant identifier and role are used to consistently assign an assessment set.

This means the same participant and role combination receives the same set.

---

# Automated Testing

The final v4 test configuration contains **42 automated tests**.

The tests cover areas such as:

- dataset counts;
- assessment-set structure;
- stable assessment assignment;
- quiz scoring;
- skip behaviour;
- deterministic concept matching;
- aliases and word variants;
- typo tolerance;
- semantic scoring;
- paraphrase recognition;
- contradiction detection;
- semantic fallback;
- role filtering;
- embedding retrieval;
- TF-IDF fallback;
- embedding cache behaviour;
- RAG prompts;
- valid citations;
- invalid citation rejection;
- score limits;
- RAG fallback behaviour;
- invitation generation;
- invitation tampering;
- invitation expiry;
- assessment integrity.

Final test output:

```text
Ran 42 tests
OK
```

This means the defined software tests passed.

It does **not** mean that the complete AI system has 100% real-world accuracy.

---

# Team Members and Responsibilities

The responsibilities below describe the main project ownership and additional implementation areas reflected in the final SkillScope AI v4 prototype.

Development and integration were collaborative, so some components involved contributions from multiple team members.

| Student ID | Full Name | Final Role / Relevant Contributions |
|---|---|---|
| **s4150922** | **Sarath Pon Kumar Murugesan** | Project coordination, system architecture, RAG integration, stakeholder requirements, business-value evaluation, experiment coordination, RAG prompt and pipeline design, structured generation validation, citation validation, fallback handling, RAG quality evaluation, and coordination of end-to-end prototype integration. |
| **s4163852** | **Dharunka Sivakumar Sasikala** | Test-Driven RAG evaluation, prompt engineering, question-bank design, ground-truth concepts and labels, assessment rubrics, evaluation metrics, quiz scoring and detailed results, semantic-scoring validation, paraphrase and contradiction testing, QA and error analysis, assessment-integrity testing, regression testing, human-review dataset preparation, reviewer-packet preparation, evaluation reproducibility, and documentation. |
| **s4177711** | **Santhiya Ravi** | Knowledge-base acquisition and curation, preprocessing, knowledge-chunk preparation, metadata design, role/skill/proficiency mapping, competency mapping, source provenance and traceability, reference-source organisation, knowledge-base quality checking, ethics and privacy considerations, and human-oversight/safe-use requirements. |
| **s4188617** | **Yaswanth Gunta** | Web application and employee assessment workflow, employee/director/HR-oriented result presentation, assessment interaction design, candidate invitation and assessment flow, consent and attempt handling, detailed result presentation, learning-recommendation display, usability testing, stakeholder experience evaluation, and interface integration with scoring and RAG outputs. |
| **s4140797** | **Satyajit Seetharaman Kavitha** | TF-IDF and semantic embedding retrieval, role-aware metadata filtering, retrieval ranking, local `nomic-embed-text` integration through Ollama, embedding caching, retrieval optimisation, hybrid-retrieval experiments, semantic retrieval stress testing, Hit@1/3/5 and MRR evaluation, backend retrieval implementation, and retrieval error analysis. |

---

# Project Structure

```text
Group-83/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── skillscope_core.py
├── semantic_scoring.py
├── rag_pipeline.py
├── web_app.py
├── run_demo.py
├── create_invitation.py
├── build_question_sets.py
│
├── evaluate_retrieval.py
├── evaluate_rag_quality.py
├── hard_semantic_retrieval_test.py
├── larger_semantic_retrieval_test.py
├── quick_embedding_hybrid_test.py
│
├── build_human_review_set.py
├── clean_human_review_set.py
├── prepare_reviewer_packets.py
├── merge_reviews_for_adjudication.py
├── final_single_reviewer_evaluation.py
│
├── test_prototype.py
├── test_rag_pipeline.py
├── test_v3_features.py
├── test_embedding_retriever.py
├── test_concept_matching.py
├── test_assessment_integrity.py
├── test_semantic_scoring.py
│
├── data/
│   ├── assessment_sets.json
│   ├── rag_quality_cases.example.jsonl
│   │
│   └── source/
│       ├── knowledge_base.jsonl
│       ├── assessment_questions.jsonl
│       ├── coding_quizzes.jsonl
│       └── evaluation_cases.jsonl
│
├── results/
│
├── RAG_ARCHITECTURE.md
├── HALLUCINATION_EVALUATION.md
├── MENTOR_UPDATE.md
└── RELEASE_NOTES_V3.md
```

Some file and document names still include `v3` because SkillScope v4 was developed incrementally from the previous prototype rather than being rebuilt as a completely separate application.

---

# Installation

## 1. Python

Python 3.11 or later is recommended.

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

The core Python requirements include:

```text
numpy
scikit-learn
```

---

# Ollama Setup

SkillScope uses Ollama for local AI models.

Install Ollama separately, then pull the required models.

## Embedding model

```bash
ollama pull nomic-embed-text
```

## Language model

```bash
ollama pull llama3.2:3b
```

Start Ollama:

```bash
ollama serve
```

Keep Ollama running while using semantic scoring, embedding retrieval, or RAG feedback.

---

# Run the Prototype

The easiest method is:

```bash
python run_demo.py
```

The demo script:

1. checks the application port;
2. starts the web application;
3. uses embedding retrieval by default;
4. creates a signed assessment invitation;
5. opens the assessment in the browser.

---

## Run for a Specific Role

Example:

```bash
python run_demo.py \
  --participant DEMO-001 \
  --role data_engineer
```

Available role codes:

```text
backend_software_engineer
frontend_ui_developer
devops_sre_engineer
qa_test_engineer
data_engineer
```

---

# Select the Retrieval Method

Embedding retrieval:

```bash
python run_demo.py --retriever embedding
```

TF-IDF baseline:

```bash
python run_demo.py --retriever tfidf
```

---

# Run the Web Application Manually

Start the application:

```bash
python web_app.py
```

Then create an invitation in another terminal:

```bash
python create_invitation.py \
  --participant EMP-001 \
  --role qa_test_engineer
```

The command prints a signed assessment URL.

Open that URL in a browser.

---

# Run the Automated Test Suite

Run:

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

Expected final v4 result:

```text
Ran 42 tests

OK
```

---

# Main Evaluation Metrics

## Retrieval

The project uses:

- Hit@1
- Hit@3
- Hit@5
- MRR@5
- Miss@5

### Hit@1

Expected evidence is the first result.

### Hit@3

Expected evidence appears within the first three results.

### Hit@5

Expected evidence appears within the first five results.

### MRR@5

Rewards systems that rank the correct evidence closer to the top.

### Miss@5

Measures how often the expected evidence is not found within the first five results.

---

# Current Limitations

The current prototype has several limitations.

## Knowledge Base

The knowledge base contains manually curated chunks and is relatively small compared with the technical knowledge used in a real organisation.

## Document Ingestion

The prototype does not yet provide an automated raw-document ingestion and chunking pipeline.

## Semantic Scoring

The language model can still classify an employee answer incorrectly.

## Competency Threshold

The 80% threshold is a prototype design choice and has not been independently validated as an industry standard.

## Retrieval Evaluation

The semantic stress tests are development experiments rather than final independently human-reviewed benchmarks.

## RAG Grounding

Citation validation confirms that citation identifiers refer to retrieved evidence, but this alone does not prove that every generated statement is fully supported.

## Human Validation

Independent technical reviewers are still required for defensible scoring and grounding measurements.

## Persistent Storage

Assessment attempts and results are currently held in application memory rather than a production database.

## Security

Production authentication, permissions, encryption, logging, and data-protection controls would need to be added before organisational deployment.

## Scalability

The prototype has not been fully tested with large numbers of simultaneous users.

---

# Future Work

Future improvements include:

- independent human validation of semantic scoring;
- concept-level precision, recall, and F1 evaluation;
- competency-agreement evaluation;
- score-error analysis;
- RAG groundedness evaluation;
- hallucination-rate evaluation;
- citation-precision evaluation;
- repeated-run consistency testing;
- automated document ingestion;
- automatic knowledge-base updating;
- larger reviewed question banks;
- persistent database storage;
- production authentication and access control;
- employee progress tracking;
- organisation-specific competency maps;
- organisation-specific technical knowledge bases;
- larger user and scalability testing;
- controlled organisational pilot studies.

---

# Human Oversight

SkillScope AI is intended to support learning decisions.

It should not be used as the only basis for:

- hiring;
- termination;
- promotion;
- remuneration;
- disciplinary action;
- formal employee ranking.

Human review remains necessary where assessment results may affect consequential decisions.

---

# Main Project Documentation

Additional documentation is available in:

### `RAG_ARCHITECTURE.md`

Describes the Retrieval-Augmented Generation architecture and evidence flow.

### `HALLUCINATION_EVALUATION.md`

Describes the proposed approach for measuring groundedness, citation quality, and hallucination behaviour.

### `MENTOR_UPDATE.md`

Contains progress and implementation updates prepared during development.

### `results/`

Contains development evaluation outputs and retrieval experiment results.

---

# Development Approach

The project followed an iterative and test-driven development process.

Important changes during development included:

1. implementing a deterministic concept-matching baseline;
2. identifying failures on valid paraphrased answers;
3. identifying false positives caused by correct keywords used in incorrect statements;
4. adding controlled semantic concept evaluation;
5. implementing role-aware TF-IDF retrieval;
6. evaluating limitations of lexical retrieval;
7. adding semantic embedding retrieval;
8. comparing TF-IDF and embedding retrieval;
9. integrating RAG learning feedback;
10. adding citation and structured-output validation;
11. expanding automated regression tests;
12. preparing human-review and future evaluation workflows.

These changes were made in response to observed prototype behaviour rather than simply adding model complexity.

---

# Repository

GitHub repository:

**https://github.com/DharunikaSivakumarSasikala/Group-83**

---

# Academic Context

This repository contains the prototype developed by **Group 83** for the Work Integrated Learning project at RMIT University.

The development metrics reported in this repository should be interpreted together with the limitations discussed in the written report.

Software test results and development retrieval experiments should not be treated as independent evidence of production-level AI accuracy.

---

# Group 83

**SkillScope AI — Role-Aware Technical Knowledge Gap Assessment and Evidence-Based Learning Support**
