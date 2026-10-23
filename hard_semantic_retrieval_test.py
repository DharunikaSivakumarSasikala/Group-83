#!/usr/bin/env python3
"""
SkillScope hard semantic retrieval experiment.

Goal
----
Test retrieval when the query expresses the SAME technical meaning using
DIFFERENT wording from the source chunk.

This script:
1. Selects 60 knowledge chunks (12 per role).
2. Uses local llama3.2:3b to generate harder paraphrases.
3. Explicitly discourages copying important source terms.
4. Measures lexical overlap and retries when the paraphrase is too similar.
5. Compares:
   - role-aware TF-IDF
   - nomic-embed-text
   - 75% TF-IDF + 25% embedding
   - 50% TF-IDF + 50% embedding
   - 25% TF-IDF + 75% embedding
6. Saves all cases and metrics under results/.

This is a development stress test, not a final human-labelled benchmark.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "source" / "knowledge_base.jsonl"
RESULTS_DIR = BASE_DIR / "results"

CASES_CACHE = RESULTS_DIR / "hard_semantic_cases.json"
OUTPUT_PATH = RESULTS_DIR / "hard_semantic_retrieval_test.json"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2:3b"

CASES_PER_ROLE = 12
TOP_K = 5
MAX_RETRIES = 4

# Lower = harder lexical stress test.
# 0.35 means at most ~35% of source content words may reappear in the query.
MAX_CONTENT_WORD_OVERLAP = 0.35

HYBRID_WEIGHTS = {
    "Hybrid_75T_25E": (0.75, 0.25),
    "Hybrid_50T_50E": (0.50, 0.50),
    "Hybrid_25T_75E": (0.25, 0.75),
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def post_json(path: str, payload: dict, timeout: int = 180) -> dict:
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


def installed_models() -> set[str]:
    tags = get_json("/api/tags")
    names = set()
    for model in tags.get("models", []):
        name = model.get("name", "")
        names.add(name)
        if name.endswith(":latest"):
            names.add(name[:-7])
    return names


def check_models() -> None:
    try:
        names = installed_models()
    except Exception as exc:
        print("ERROR: Ollama is not reachable at http://127.0.0.1:11434")
        print(f"Details: {exc}")
        sys.exit(1)

    missing = [m for m in (EMBED_MODEL, GEN_MODEL) if m not in names]
    if missing:
        print("Missing Ollama model(s):", ", ".join(missing))
        for model in missing:
            print(f"  ollama pull {model}")
        sys.exit(1)


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_words(text: str) -> list[str]:
    words = []
    for t in tokens(text):
        if len(t) < 3:
            continue
        if t in ENGLISH_STOP_WORDS:
            continue
        words.append(t)
    return words


def overlap_ratio(source: str, query: str) -> float:
    src = set(content_words(source))
    qry = set(content_words(query))
    if not src:
        return 0.0
    return len(src & qry) / len(src)


def important_terms(text: str, limit: int = 12) -> list[str]:
    words = content_words(text)
    counts = Counter(words)

    # Prefer repeated/longer terms because they tend to be more diagnostic.
    scored = sorted(
        counts,
        key=lambda w: (counts[w], len(w)),
        reverse=True,
    )
    return scored[:limit]


def clean_generated_query(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(
        r"^\s*(question|query|paraphrase)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    text = " ".join(text.split())
    return text


def generate_hard_query(chunk: dict, attempt: int) -> str:
    banned = important_terms(
        f"{chunk.get('skill', '')} {chunk.get('text', '')}",
        limit=12,
    )
    banned_text = ", ".join(banned)

    extra = ""
    if attempt > 1:
        extra = (
            "\nPrevious wording was still too lexically similar. "
            "Use more everyday synonyms and restructure the sentence substantially."
        )

    prompt = f"""
Create ONE difficult semantic retrieval query from the technical passage below.

Technical role:
{chunk['role']}

Source passage:
{chunk['text']}

The query MUST preserve the same technical meaning, but it should be hard for
a keyword-only search system.

Avoid reusing these source terms where reasonably possible:
{banned_text}

Rules:
- Express the idea using different wording and synonyms.
- Do not copy phrases from the passage.
- Do not mention the chunk ID, source title, document title, or this task.
- Do not answer the question.
- Keep unavoidable product/protocol names only when meaning would become unclear.
- Prefer natural employee wording rather than textbook wording.
- Output only ONE sentence.
{extra}
""".strip()

    response = post_json(
        "/api/generate",
        {
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
    )
    query = clean_generated_query(response.get("response", ""))
    if not query:
        raise RuntimeError(f"No query generated for {chunk['chunk_id']}")
    return query


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if len(items) <= count:
        return items[:]

    raw = np.linspace(0, len(items) - 1, count)
    indices = []

    for value in raw:
        idx = int(round(float(value)))
        if idx not in indices:
            indices.append(idx)

    for idx in range(len(items)):
        if len(indices) >= count:
            break
        if idx not in indices:
            indices.append(idx)

    return [items[i] for i in sorted(indices[:count])]


def build_or_load_cases(chunks: list[dict]) -> list[dict]:
    RESULTS_DIR.mkdir(exist_ok=True)

    if CASES_CACHE.exists():
        cases = json.loads(CASES_CACHE.read_text(encoding="utf-8"))
        print(f"Using cached hard semantic cases: {CASES_CACHE}")
        return cases

    by_role: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_role[chunk["role"]].append(chunk)

    cases = []

    print("Generating HARD low-overlap semantic cases...")
    print(
        f"Target maximum source-word overlap: "
        f"{MAX_CONTENT_WORD_OVERLAP:.0%}"
    )

    for role in sorted(by_role):
        selected = evenly_spaced(by_role[role], CASES_PER_ROLE)
        print(f"\n{role}: {len(selected)} cases")

        for chunk in selected:
            best_query = None
            best_overlap = 1.0
            accepted = False

            for attempt in range(1, MAX_RETRIES + 1):
                query = generate_hard_query(chunk, attempt)
                overlap = overlap_ratio(chunk["text"], query)

                if overlap < best_overlap:
                    best_query = query
                    best_overlap = overlap

                if overlap <= MAX_CONTENT_WORD_OVERLAP:
                    accepted = True
                    break

            cases.append(
                {
                    "case_id": f"HARD_{chunk['chunk_id']}",
                    "role": role,
                    "expected_chunk_id": chunk["chunk_id"],
                    "skill": chunk.get("skill", ""),
                    "query": best_query,
                    "content_word_overlap": best_overlap,
                    "overlap_target_met": accepted,
                }
            )

            mark = "OK" if accepted else "KEEP-BEST"
            print(
                f"  {chunk['chunk_id']}: "
                f"overlap={best_overlap:.2%} [{mark}]"
            )
            print(f"    {best_query}")

    CASES_CACHE.write_text(
        json.dumps(cases, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved hard semantic cases to: {CASES_CACHE}")
    return cases


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
            raise RuntimeError("Ollama returned no embeddings.")
        vectors.extend(batch_vectors)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def minmax_for_candidates(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(scores, dtype=np.float64)
    values = scores[candidate_indices]
    lo = float(values.min())
    hi = float(values.max())

    if math.isclose(lo, hi):
        result[candidate_indices] = 0.0
    else:
        result[candidate_indices] = (values - lo) / (hi - lo)

    return result


def rank_expected(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    ids: list[str],
    expected: str,
) -> tuple[int, list[str]]:
    ordered = candidate_indices[
        np.argsort(-scores[candidate_indices], kind="mergesort")
    ]
    ranked = [ids[i] for i in ordered]
    return ranked.index(expected) + 1, ranked[:TOP_K]


def metric_summary(ranks: list[int]) -> dict[str, float | int]:
    n = len(ranks)
    return {
        "cases": n,
        "hit_at_1": sum(r <= 1 for r in ranks) / n,
        "hit_at_3": sum(r <= 3 for r in ranks) / n,
        "hit_at_5": sum(r <= 5 for r in ranks) / n,
        "mrr_at_5": sum(
            (1.0 / r if r <= 5 else 0.0)
            for r in ranks
        ) / n,
        "miss_rate_at_5": sum(r > 5 for r in ranks) / n,
        "misses_at_5": sum(r > 5 for r in ranks),
    }


def pct(value: float) -> str:
    return f"{100 * value:6.2f}%"


def main() -> None:
    if not KB_PATH.exists():
        print(f"ERROR: Could not find {KB_PATH}")
        print("Place this file in SkillScope_AI_Prototype.")
        sys.exit(1)

    check_models()

    chunks = load_jsonl(KB_PATH)
    ids = [c["chunk_id"] for c in chunks]
    roles = [c["role"] for c in chunks]
    documents = [
        f"{c['skill']} {c['level']} {c['text']}"
        for c in chunks
    ]

    cases = build_or_load_cases(chunks)

    overlaps = [c["content_word_overlap"] for c in cases]
    passed_overlap = sum(
        c["overlap_target_met"]
        for c in cases
    )

    print("\n" + "=" * 90)
    print("SKILLSCOPE HARD SEMANTIC RETRIEVAL TEST")
    print("=" * 90)
    print(f"Knowledge chunks         : {len(chunks)}")
    print(f"Hard semantic cases      : {len(cases)}")
    print(f"Cases per role           : {CASES_PER_ROLE}")
    print(f"Embedding model          : {EMBED_MODEL}")
    print(f"Role filtering           : enabled")
    print(
        f"Mean source-word overlap : "
        f"{np.mean(overlaps):.2%}"
    )
    print(
        f"Median overlap           : "
        f"{np.median(overlaps):.2%}"
    )
    print(
        f"Cases meeting <= {MAX_CONTENT_WORD_OVERLAP:.0%} overlap: "
        f"{passed_overlap}/{len(cases)}"
    )
    print()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    t0 = time.perf_counter()
    doc_tfidf = vectorizer.fit_transform(documents)
    tfidf_index_seconds = time.perf_counter() - t0

    print("Embedding knowledge chunks...")
    t0 = time.perf_counter()
    doc_embeddings = embed_texts(documents)
    embedding_index_seconds = time.perf_counter() - t0

    queries = [c["query"] for c in cases]

    t0 = time.perf_counter()
    query_tfidf = vectorizer.transform(queries)
    tfidf_query_seconds = time.perf_counter() - t0

    print("Embedding hard semantic queries...")
    t0 = time.perf_counter()
    query_embeddings = embed_texts(queries)
    embedding_query_seconds = time.perf_counter() - t0

    method_names = [
        "TF-IDF",
        "Embedding",
        *HYBRID_WEIGHTS.keys(),
    ]

    ranks: dict[str, list[int]] = {
        name: []
        for name in method_names
    }

    role_ranks: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {
            name: []
            for name in method_names
        }
    )

    detailed = []

    for case_index, case in enumerate(cases):
        role = case["role"]
        expected = case["expected_chunk_id"]

        candidate_indices = np.asarray(
            [
                i
                for i, chunk_role in enumerate(roles)
                if chunk_role == role
            ],
            dtype=int,
        )

        tfidf_scores = (
            doc_tfidf @ query_tfidf[case_index].T
        ).toarray().ravel()

        embedding_scores = (
            doc_embeddings @ query_embeddings[case_index]
        )

        tfidf_norm = minmax_for_candidates(
            tfidf_scores,
            candidate_indices,
        )
        embedding_norm = minmax_for_candidates(
            embedding_scores,
            candidate_indices,
        )

        scores_by_method = {
            "TF-IDF": tfidf_scores,
            "Embedding": embedding_scores,
        }

        for name, (tw, ew) in HYBRID_WEIGHTS.items():
            scores_by_method[name] = (
                tw * tfidf_norm
                + ew * embedding_norm
            )

        row = {
            "case_id": case["case_id"],
            "role": role,
            "skill": case["skill"],
            "expected_chunk_id": expected,
            "query": case["query"],
            "content_word_overlap": case["content_word_overlap"],
            "methods": {},
        }

        for method, scores in scores_by_method.items():
            rank, top5 = rank_expected(
                scores,
                candidate_indices,
                ids,
                expected,
            )
            ranks[method].append(rank)
            role_ranks[role][method].append(rank)
            row["methods"][method] = {
                "rank": rank,
                "top5": top5,
            }

        detailed.append(row)

    summaries = {
        name: metric_summary(ranks[name])
        for name in method_names
    }

    print("\nOVERALL RESULTS")
    print("-" * 90)
    print(
        f"{'Method':<18}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'MRR@5':>10}"
        f"{'Miss@5':>10}"
        f"{'Misses':>8}"
    )

    for name in method_names:
        m = summaries[name]
        print(
            f"{name:<18}"
            f"{pct(m['hit_at_1']):>10}"
            f"{pct(m['hit_at_3']):>10}"
            f"{pct(m['hit_at_5']):>10}"
            f"{m['mrr_at_5']:>10.4f}"
            f"{pct(m['miss_rate_at_5']):>10}"
            f"{m['misses_at_5']:>8}"
        )

    print("\nRESULTS BY ROLE — Hit@5 / MRR@5")
    print("-" * 90)

    for role in sorted(role_ranks):
        print(f"\n{role}")
        for name in method_names:
            m = metric_summary(
                role_ranks[role][name]
            )
            print(
                f"  {name:<18} "
                f"Hit@5={pct(m['hit_at_5'])}  "
                f"MRR@5={m['mrr_at_5']:.4f}  "
                f"misses={m['misses_at_5']}"
            )

    print("\nBIGGEST EMBEDDING GAINS OVER TF-IDF")
    print("-" * 90)

    gains = []
    for row in detailed:
        tf_rank = row["methods"]["TF-IDF"]["rank"]
        em_rank = row["methods"]["Embedding"]["rank"]
        gains.append(
            (
                tf_rank - em_rank,
                row["expected_chunk_id"],
                tf_rank,
                em_rank,
                row["content_word_overlap"],
            )
        )

    gains.sort(reverse=True)

    for gain, chunk_id, tf_rank, em_rank, overlap in gains[:12]:
        if gain <= 0:
            break
        print(
            f"{chunk_id:<12} "
            f"TF-IDF={tf_rank:<3} "
            f"Embedding={em_rank:<3} "
            f"gain={gain:<3} "
            f"overlap={overlap:.1%}"
        )

    print("\nTIMING (batch experiment)")
    print("-" * 90)
    print(f"TF-IDF index build       : {tfidf_index_seconds:.4f} s")
    print(f"Embedding document index : {embedding_index_seconds:.4f} s")
    print(
        f"TF-IDF queries           : {tfidf_query_seconds:.4f} s "
        f"({1000 * tfidf_query_seconds / len(cases):.2f} ms/query)"
    )
    print(
        f"Embedding queries        : {embedding_query_seconds:.4f} s "
        f"({1000 * embedding_query_seconds / len(cases):.2f} ms/query)"
    )

    best = max(
        method_names,
        key=lambda name: (
            summaries[name]["hit_at_5"],
            summaries[name]["mrr_at_5"],
            summaries[name]["hit_at_1"],
        ),
    )

    print("\nINTERPRETATION")
    print("-" * 90)
    print(f"Best by Hit@5, then MRR@5: {best}")
    print(
        "Use this as semantic-stress evidence only. "
        "Final deployment choice still needs human-reviewed held-out cases."
    )

    report = {
        "experiment": {
            "knowledge_chunks": len(chunks),
            "semantic_cases": len(cases),
            "cases_per_role": CASES_PER_ROLE,
            "top_k": TOP_K,
            "max_content_word_overlap": MAX_CONTENT_WORD_OVERLAP,
            "mean_content_word_overlap": float(np.mean(overlaps)),
            "median_content_word_overlap": float(np.median(overlaps)),
            "cases_meeting_overlap_target": passed_overlap,
            "embedding_model": EMBED_MODEL,
            "query_generator": GEN_MODEL,
            "hybrid_weights": {
                key: {
                    "tfidf": value[0],
                    "embedding": value[1],
                }
                for key, value in HYBRID_WEIGHTS.items()
            },
        },
        "metrics": summaries,
        "metrics_by_role": {
            role: {
                method: metric_summary(
                    role_ranks[role][method]
                )
                for method in method_names
            }
            for role in sorted(role_ranks)
        },
        "timing_seconds": {
            "tfidf_index_build": tfidf_index_seconds,
            "embedding_document_index": embedding_index_seconds,
            "tfidf_all_queries": tfidf_query_seconds,
            "embedding_all_queries": embedding_query_seconds,
        },
        "cases": detailed,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved detailed results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
