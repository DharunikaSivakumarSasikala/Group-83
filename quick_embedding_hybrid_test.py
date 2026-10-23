#!/usr/bin/env python3
"""
Quick semantic retrieval smoke test for SkillScope AI.

Compares:
1) role-aware TF-IDF baseline
2) local Ollama text embeddings (default: nomic-embed-text)
3) hybrid retrieval = normalized TF-IDF + embedding cosine similarity

This is intentionally SMALL: 15 semantic/paraphrase cases, 3 per role.
It does not modify the SkillScope application or datasets.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "source" / "knowledge_base.jsonl"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 5

# 15 deliberately paraphrased semantic cases:
# 3 per role, with the expected evidence chunk identified in advance.
CASES = [
    {
        "role": "backend_software_engineer",
        "expected": "BACK_006",
        "query": "How do we make sure a signed-in user is allowed to read or change this specific record?",
    },
    {
        "role": "backend_software_engineer",
        "expected": "BACK_010",
        "query": "Information coming from an outside service should not be trusted automatically before our application uses it.",
    },
    {
        "role": "backend_software_engineer",
        "expected": "BACK_020",
        "query": "When a function stops early because of an error, how should opened files and allocated memory be safely released?",
    },
    {
        "role": "frontend_ui_developer",
        "expected": "FRON_003",
        "query": "What should a screen reader receive for an informative picture, and what about an image that is only decoration?",
    },
    {
        "role": "frontend_ui_developer",
        "expected": "FRON_004",
        "query": "A user who cannot use a mouse still needs to operate every control and see where navigation currently is.",
    },
    {
        "role": "frontend_ui_developer",
        "expected": "FRON_007",
        "query": "How can a screen reader be informed when page content changes without reloading the page?",
    },
    {
        "role": "devops_sre_engineer",
        "expected": "DEVO_003",
        "query": "Pods may be replaced, but clients still need one dependable network address to reach the application.",
    },
    {
        "role": "devops_sre_engineer",
        "expected": "DEVO_005",
        "query": "Is encoding a password-like value as base64 enough to protect it in a Kubernetes cluster?",
    },
    {
        "role": "devops_sre_engineer",
        "expected": "DEVO_013",
        "query": "How can monitoring automatically raise an alert when a measured condition remains true?",
    },
    {
        "role": "qa_test_engineer",
        "expected": "QA_T_004",
        "query": "Which element selectors are less likely to break when the page markup is refactored?",
    },
    {
        "role": "qa_test_engineer",
        "expected": "QA_T_010",
        "query": "How should temporary test resources be cleaned up even when the test throws an exception?",
    },
    {
        "role": "qa_test_engineer",
        "expected": "QA_T_014",
        "query": "How can we reduce tests that randomly fail because the UI was not ready yet?",
    },
    {
        "role": "data_engineer",
        "expected": "DATA_004",
        "query": "What is the proper Airflow mechanism for passing a small piece of information from one task to another?",
    },
    {
        "role": "data_engineer",
        "expected": "DATA_007",
        "query": "If a failed pipeline step is run again, how can we avoid creating duplicate records?",
    },
    {
        "role": "data_engineer",
        "expected": "DATA_012",
        "query": "What should we record so we can trace which upstream datasets and tasks produced an output table?",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def post_json(path: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(f"{OLLAMA_URL}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_ollama_model() -> None:
    try:
        tags = get_json("/api/tags")
    except Exception as exc:
        print("\nERROR: Ollama is not reachable at http://127.0.0.1:11434")
        print("Make sure Ollama is running, then try again.")
        print(f"Details: {exc}")
        sys.exit(1)

    names = {m.get("name", "") for m in tags.get("models", [])}
    if EMBED_MODEL not in names and f"{EMBED_MODEL}:latest" not in names:
        print(f"\nEmbedding model '{EMBED_MODEL}' is not installed.")
        print("Run this once:")
        print(f"  ollama pull {EMBED_MODEL}")
        sys.exit(1)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = post_json(
            "/api/embed",
            {"model": EMBED_MODEL, "input": batch},
        )
        batch_vectors = response.get("embeddings")
        if not batch_vectors:
            raise RuntimeError(
                "Ollama returned no embeddings. "
                "Check that your Ollama version supports /api/embed."
            )
        vectors.extend(batch_vectors)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def minmax_role(scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.zeros_like(scores, dtype=np.float64)
    values = scores[indices]
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        result[indices] = 0.0
    else:
        result[indices] = (values - lo) / (hi - lo)
    return result


def rank_expected(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    ids: list[str],
    expected_id: str,
) -> tuple[int, list[str]]:
    order = candidate_indices[
        np.argsort(-scores[candidate_indices], kind="mergesort")
    ]
    ranked_ids = [ids[i] for i in order]
    return ranked_ids.index(expected_id) + 1, ranked_ids[:TOP_K]


def metrics(ranks: list[int]) -> dict[str, float]:
    n = len(ranks)
    return {
        "Hit@1": sum(r <= 1 for r in ranks) / n,
        "Hit@3": sum(r <= 3 for r in ranks) / n,
        "Hit@5": sum(r <= 5 for r in ranks) / n,
        "MRR@5": sum((1.0 / r if r <= 5 else 0.0) for r in ranks) / n,
        "Miss@5": sum(r > 5 for r in ranks) / n,
    }


def pct(v: float) -> str:
    return f"{100*v:6.2f}%"


def main() -> None:
    if not KB_PATH.exists():
        print(f"ERROR: Could not find {KB_PATH}")
        print("Place this script in the SkillScope_AI_Prototype folder.")
        sys.exit(1)

    check_ollama_model()

    chunks = load_jsonl(KB_PATH)
    ids = [c["chunk_id"] for c in chunks]
    roles = [c["role"] for c in chunks]
    documents = [
        f"{c['skill']} {c['level']} {c['text']}"
        for c in chunks
    ]

    print("=" * 72)
    print("SKILLSCOPE QUICK SEMANTIC RETRIEVAL TEST")
    print("=" * 72)
    print(f"Knowledge chunks : {len(chunks)}")
    print(f"Test cases       : {len(CASES)}")
    print(f"Embedding model  : {EMBED_MODEL}")
    print("Candidate filter : same technical role")
    print("Hybrid weights   : 50% TF-IDF + 50% embedding")
    print()

    # Baseline sparse representation.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    doc_tfidf = vectorizer.fit_transform(documents)

    # True dense embeddings from local Ollama model.
    print("Embedding knowledge chunks...")
    doc_embeddings = embed_texts(documents)
    print("Embedding test queries...")
    query_embeddings = embed_texts([c["query"] for c in CASES])

    ranks = {
        "TF-IDF": [],
        "Embedding": [],
        "Hybrid": [],
    }

    case_rows = []

    for case_index, case in enumerate(CASES):
        role = case["role"]
        query = case["query"]
        expected = case["expected"]

        candidate_indices = np.asarray(
            [i for i, r in enumerate(roles) if r == role],
            dtype=int,
        )

        # TF-IDF cosine similarity. sklearn TF-IDF vectors are L2-normalized.
        query_tfidf = vectorizer.transform([query])
        tfidf_scores = (doc_tfidf @ query_tfidf.T).toarray().ravel()

        # Embedding cosine similarity. Vectors were normalized above.
        embedding_scores = doc_embeddings @ query_embeddings[case_index]

        # Normalize each score family within the role before blending.
        tfidf_norm = minmax_role(tfidf_scores, candidate_indices)
        embedding_norm = minmax_role(embedding_scores, candidate_indices)
        hybrid_scores = 0.50 * tfidf_norm + 0.50 * embedding_norm

        tf_rank, tf_top = rank_expected(
            tfidf_scores, candidate_indices, ids, expected
        )
        em_rank, em_top = rank_expected(
            embedding_scores, candidate_indices, ids, expected
        )
        hy_rank, hy_top = rank_expected(
            hybrid_scores, candidate_indices, ids, expected
        )

        ranks["TF-IDF"].append(tf_rank)
        ranks["Embedding"].append(em_rank)
        ranks["Hybrid"].append(hy_rank)

        case_rows.append(
            {
                "expected": expected,
                "tfidf_rank": tf_rank,
                "embedding_rank": em_rank,
                "hybrid_rank": hy_rank,
                "tfidf_top5": tf_top,
                "embedding_top5": em_top,
                "hybrid_top5": hy_top,
            }
        )

    print("\nSUMMARY")
    print("-" * 72)
    print(
        f"{'Method':<14}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'MRR@5':>10}"
        f"{'Miss@5':>10}"
    )

    summaries = {}
    for name in ("TF-IDF", "Embedding", "Hybrid"):
        m = metrics(ranks[name])
        summaries[name] = m
        print(
            f"{name:<14}"
            f"{pct(m['Hit@1']):>10}"
            f"{pct(m['Hit@3']):>10}"
            f"{pct(m['Hit@5']):>10}"
            f"{m['MRR@5']:>10.4f}"
            f"{pct(m['Miss@5']):>10}"
        )

    print("\nPER-CASE RANKS (lower is better)")
    print("-" * 72)
    print(f"{'Expected':<12}{'TF-IDF':>10}{'Embed':>10}{'Hybrid':>10}")
    for row in case_rows:
        print(
            f"{row['expected']:<12}"
            f"{row['tfidf_rank']:>10}"
            f"{row['embedding_rank']:>10}"
            f"{row['hybrid_rank']:>10}"
        )

    print("\nCASES WHERE EMBEDDING/HYBRID IMPROVED OVER TF-IDF")
    print("-" * 72)
    improved = 0
    for row in case_rows:
        best_new = min(row["embedding_rank"], row["hybrid_rank"])
        if best_new < row["tfidf_rank"]:
            improved += 1
            print(
                f"{row['expected']}: "
                f"TF-IDF rank {row['tfidf_rank']} -> "
                f"embedding {row['embedding_rank']}, "
                f"hybrid {row['hybrid_rank']}"
            )
    if improved == 0:
        print("None in this 15-case smoke test.")

    print("\nINTERPRETATION RULE")
    print("-" * 72)
    tf = summaries["TF-IDF"]
    em = summaries["Embedding"]
    hy = summaries["Hybrid"]

    if hy["Hit@5"] > tf["Hit@5"] and hy["MRR@5"] >= tf["MRR@5"]:
        print(
            "PROMISING: hybrid improved top-5 coverage without reducing MRR@5. "
            "Run the larger held-out evaluation next."
        )
    elif em["Hit@5"] > tf["Hit@5"]:
        print(
            "PROMISING FOR EMBEDDINGS: dense retrieval improved top-5 coverage. "
            "Tune hybrid weights before scaling."
        )
    else:
        print(
            "NO CLEAR WIN YET: do not replace TF-IDF based on this smoke test. "
            "Inspect failed cases and/or try another embedding model."
        )

    output = {
        "model": EMBED_MODEL,
        "test_case_count": len(CASES),
        "metrics": summaries,
        "ranks": ranks,
        "cases": case_rows,
    }

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "quick_embedding_hybrid_test.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
