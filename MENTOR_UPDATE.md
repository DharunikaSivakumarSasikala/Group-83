# SkillScope AI — Weekly Mentor Update

## What we completed

We moved from the project proposal into a working baseline prototype. It now:

- loads the 120-chunk knowledge base covering five technical roles with OTT content;
- creates five assessment sets per role, giving 25 sets in total;
- assigns a participant to a stable set using an anonymous participant ID;
- accepts written answers through a local browser interface;
- retrieves role-relevant technical evidence;
- produces transparent draft scores and learning gaps for human review; and
- runs locally without a paid API.

## First measured result

| Retrieval configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Miss rate @5 |
|---|---:|---:|---:|---:|---:|
| Global TF-IDF | 42.17% | 50.60% | 53.01% | 46.47% | 46.99% |
| Role-aware TF-IDF | 44.58% | 51.81% | 61.45% | 49.92% | 38.55% |

Filtering by the employee's role improved Hit@5 by 8.44 percentage points. This supports the role-aware architecture, while the remaining misses show that TF-IDF alone is not enough.

## What to demonstrate

1. Start the browser application.
2. Enter anonymous ID `DEMO-001`.
3. Select `QA and Testing`.
4. Accept the consent notice and begin.
5. Answer the first question and show the fixed set ID.
6. Complete the questions and show the evidence-linked draft result.
7. Explain that a technical director reviews the result before it becomes final.

## What we learned

- Role metadata improves retrieval accuracy.
- Keyword retrieval struggles when the question and supporting passage use different wording.
- The current 83-question bank can create 25 ten-question sets without repetition inside one set, but some questions overlap across different sets.
- A deterministic participant-to-set assignment prevents refreshes from changing the test.

## Next experiment

Add dense semantic embeddings and compare them with the TF-IDF baseline using the same held-out questions. Then add reranking and local Ollama-based evidence-grounded scoring. The baseline metrics above will make the improvement measurable.

## Short spoken update

> This week we moved from planning to a working prototype. The system now loads our five-role OTT knowledge base, creates 25 assessment sets, assigns a stable set using an anonymous participant ID, accepts employee answers and retrieves evidence for a draft learning-gap result. We also measured the first baseline. Role-aware TF-IDF achieved 61.45 percent Hit@5 compared with 53.01 percent without role filtering. This shows that role metadata helps, but it also gives us a clear next task: semantic retrieval and reranking. Everything currently runs locally with no paid API.
