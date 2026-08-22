# SkillScope AI

**Group 83 — COSC2650 Case Studies in Data Science, 2026**  
**RMIT University — Master of Data Science**

## Project Aim

SkillScope AI aims to develop and evaluate a role-aware Retrieval-Augmented Generation (RAG) system that identifies evidence-based technical knowledge gaps across software teams and provides personalised learning recommendations grounded in authoritative technical documentation.

## Team Members

| Student ID | Full Name | Proposed Role / Relevant Skills |
|---|---|---|
| s4150922 | Sarath Pon Kumar Murugesan | Project coordination, system architecture, RAG integration, stakeholder requirements, business-value evaluation, experiment coordination |
| s4163852 | Dharunka Sivakumar Sasikala | Test-Driven RAG evaluation, prompt engineering, question-bank design, ground-truth labels, assessment rubrics, evaluation metrics, QA, error analysis, experiment reproducibility |
| s4177711 | Santhiya Ravi | Knowledge-base acquisition, preprocessing, chunking, metadata design, competency mapping, source provenance, ethics and privacy |
| s4188617 | Yaswanth Gunta | Streamlit application, employee/director/HR views, learning recommendation presentation, usability testing, stakeholder experience evaluation |
| s4140797 | Satyajit Seetharaman Kavitha | BM25 and embedding retrieval, vector database, metadata filtering, reranking, Ollama/LLM integration, backend implementation, retrieval optimisation |

## User Stories

- **As a technical employee, I want to complete a role-specific knowledge assessment so that I can identify the technical areas in which I need further development.**
- **As a technical employee, I want to receive personalised learning recommendations based on my identified knowledge gaps so that I can focus my learning on the most relevant technical topics.**
- **As a technical director or team lead, I want to view evidence-linked competency results so that I can understand employee development needs and prioritise suitable training.**
- **As an HR or Learning and Development stakeholder, I want to view an approved summary of employee knowledge gaps so that I can plan targeted training programs without needing to interpret detailed technical assessment responses.**

## Initial Technical Roles

1. Backend Engineering
2. Frontend/UI Development
3. DevOps/SRE
4. QA/Testing
5. Data Engineering

## Proposed RAG Workflow

`Technical Documents → Chunking & Metadata → Retrieval → Employee Assessment → Knowledge Gap Detection → Learning Recommendation`

The project will compare keyword/BM25 retrieval with embedding-based retrieval, metadata filtering and reranking. Grounded generation is planned through a locally hosted LLM using Ollama.

## Test-Driven RAG Evaluation

Planned evaluation includes:

- Retrieval effectiveness: Hit@1/3/5, MRR and/or NDCG
- Grounding: faithfulness and citation correctness
- Assessment value: rubric agreement and knowledge-gap classification F1
- Stakeholder value: recommendation relevance and usability
- Safety: appropriate abstention for out-of-knowledge-base questions

The project also reproduces the Walert RAG evaluation approach as an initial methodological validation before applying the evaluation framework to SkillScope AI.

## Repository Structure

```text
skillscope-ai/
├── README.md
├── app/          # Streamlit application
├── data/         # Evaluation data and processed corpus metadata
├── docs/         # Project documentation
├── evaluation/   # Retrieval/generation evaluation scripts and results
├── src/          # RAG, retrieval and backend source code
└── tests/        # Unit and integration tests
```

## Project Status

Milestone 1: project scope, stakeholders, competency areas, knowledge-base plan, metadata design, and preliminary Walert evaluation workflow defined. SkillScope-specific RAG implementation and evaluation are the next development stages.

## Trello

Add the teaching-team-accessible Trello board link here.

## GitHub Access

This repository is intended to remain **private**. Access should be granted to the assigned mentor/teaching team when requested.
# Group-83
