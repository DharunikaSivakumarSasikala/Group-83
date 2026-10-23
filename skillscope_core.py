"""Core retrieval, set allocation, and rubric scoring for SkillScope AI."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "source"
COMPETENCY_THRESHOLD = 80.0


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reference_chunk_ids(question: dict) -> list[str]:
    """Return all approved reference chunks for a written question.

    Most questions have one primary reference. A small number legitimately need
    two approved chunks to cover the complete rubric. Keeping this helper in the
    core prevents the UI and evaluation scripts from disagreeing about what
    counts as approved evidence.
    """
    values = question.get("reference_chunk_ids")
    if values:
        return [str(value) for value in values if str(value).strip()]
    primary = question.get("reference_chunk_id")
    return [str(primary)] if primary else []


def normalise(text: str) -> str:
    """Normalise user text without changing its technical meaning."""
    text = text.lower().replace("–", "-").replace("—", "-")
    # Hyphens usually separate natural-language words in the rubric
    # (for example user-facing, start-up, non-confidential).
    text = text.replace("-", " ")
    return re.sub(r"[^a-z0-9+#.]+", " ", text).strip()


# Small transparent derivational families used by the deterministic scorer.
# These cover common technical wording changes without asking the LLM to score.
_DERIVATION_GROUPS = (
    ("accessibility", "accessible", "accessibly"),
    ("allocate", "allocation", "allocated", "allocating"),
    ("announce", "announced", "announcement", "announcements"),
    ("authorize", "authorized", "authorization"),
    ("cache", "cached", "caching"),
    ("change", "changes", "changed", "changing"),
    ("cancel", "cancelled", "canceled", "cancellation"),
    ("configure", "configured", "configuration"),
    ("corrupt", "corrupted", "corruption"),
    ("destroy", "destroyed", "destruction"),
    ("diagnose", "diagnosed", "diagnosis"),
    ("duplicate", "duplicated", "duplication"),
    ("fail", "fails", "failed", "failure", "failures"),
    ("global", "globally"),
    ("idempotent", "idempotence", "idempotency"),
    ("impact", "impacts", "impacted", "impacting"),
    ("operate", "operable", "operation", "operations"),
    ("reliable", "reliability"),
    ("reuse", "reused", "reusable", "reusing"),
    ("select", "selected", "selecting", "selection", "selectable"),
    ("synchronize", "synchronise", "synchronized", "synchronised", "synchronization", "synchronisation"),
    ("validate", "validated", "validating", "validation"),
)

_DERIVATION_CANONICAL: dict[str, str] = {}
for _group in _DERIVATION_GROUPS:
    _canonical = _group[0]
    for _form in _group:
        _DERIVATION_CANONICAL[_form] = _canonical


def _singularise(token: str) -> str:
    """Conservative English singularisation for rubric tokens."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _canonical_token(token: str) -> str:
    token = normalise(token).strip(".")
    if not token:
        return ""

    # Exact derivational family first, then singular form and family again.
    if token in _DERIVATION_CANONICAL:
        return _DERIVATION_CANONICAL[token]
    token = _singularise(token)
    return _DERIVATION_CANONICAL.get(token, token)


def _stem(token: str) -> str:
    """Backward-compatible name for the deterministic canonicaliser."""
    return _canonical_token(token)


def concept_tokens(text: str) -> set[str]:
    return {
        _canonical_token(token)
        for token in normalise(text).split()
        if _canonical_token(token)
    }


# Controlled, auditable paraphrases accepted by the deterministic scorer.
# Every semantic alternative remains explicit and reviewable.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "key": (
        "message key", "message keys", "record key", "record keys",
        "partition key", "partition keys",
    ),
    "duplicate": (
        "more than once", "arrives more than once", "arriving more than once",
        "repeated event", "repeated events", "same event twice",
    ),
    "late": (
        "later than expected", "arrives late", "arrive late", "arriving late",
        "delayed event", "delayed events",
    ),
    "malformed": (
        "badly formed", "invalid format", "invalid event format",
        "invalid event formats",
    ),
    "counts": (
        "record count", "record counts", "row count", "row counts",
        "number of records", "number of rows",
    ),
    "rejected records": (
        "rejected rows", "failed records", "failed rows",
        "records that were rejected", "rows that were rejected",
    ),
    "field mappings": (
        "field mapping", "mapped fields", "column mapping", "column mappings",
        "mapped columns",
    ),
    "samples": (
        "sample records", "sample rows", "representative records",
        "representative rows", "representative sample", "representative samples",
    ),
    "meaning": ("semantics", "semantic meaning"),
    "order": ("sequence", "navigation sequence", "dependency order"),
    "context": ("contextual", "contextual requirements"),
    "frequency": ("how often", "how frequently"),
    "startup": ("start up", "start-up"),
    "endpoint": ("stable abstraction", "stable network abstraction"),
    "cost": ("paid service", "paid services", "paid downstream service", "paid downstream services"),
    "no hidden controls": ("without hiding controls", "do not hide controls", "controls should not be hidden"),
    "reflow": ("constrained viewport", "small viewport", "without two dimensional scrolling", "without unnecessary two dimensional scrolling"),
    "services": ("multiple containers", "one or more containers", "containers run together"),
    "run": ("runtime", "run together", "running"),
    "external": ("externally", "stored externally"),
    "reference": ("referenced", "referenced by", "location or identifier"),
    "idempotent": (
        "safe rerun", "safe reruns", "rerun safely", "repeated processing",
        "reruns do not duplicate", "rerun does not duplicate",
        "reruns do not duplicate content",
    ),
    "latency": (
        "takes longer than expected", "takes much longer than expected",
        "longer than expected", "delayed completion", "slow completion",
        "breached latency threshold", "breached latency thresholds",
    ),
    "actionable": ("requires action", "needs action", "actionable symptom", "actionable symptoms"),
    "desired state": ("declared state", "intended state"),
    "accessible announcement": ("announced accessibly", "accessible notification", "announced to assistive technology"),
    "selectable": ("can be selected", "available for selection", "expose available tracks"),
    "user facing": ("role label or visible text", "role or label", "visible text"),
    "resilient": ("robust", "less brittle"),
    "unbounded": ("not bounded", "uncontrolled labels", "unbounded labels"),
    "cardinality": ("too many time series", "cardinality growth", "high cardinality"),
}


def _tokens_match(expected_text: str, answer_tokens: set[str]) -> bool:
    expected = concept_tokens(expected_text)
    return bool(expected) and expected.issubset(answer_tokens)


def _inflections(word: str) -> set[str]:
    """Generate a few common inflections for typo-tolerant comparison."""
    word = normalise(word)
    if not word or " " in word:
        return {word}
    forms = {word}
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        forms.update({word[:-1] + "ies", word[:-1] + "ied", word + "ing"})
    elif word.endswith("e"):
        forms.update({word + "s", word + "d", word[:-1] + "ing"})
    else:
        forms.update({word + "s", word + "ed", word + "ing"})
    return forms


def _damerau_distance_at_most_one(a: str, b: str) -> bool:
    """True for exact match or one insertion/deletion/substitution/transposition."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return j == i + 1 and a[i] == b[j] and a[j] == b[i]
        return False
    # One insertion/deletion.
    short, long = (a, b) if len(a) < len(b) else (b, a)
    i = j = mismatches = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
        else:
            mismatches += 1
            if mismatches > 1:
                return False
            j += 1
    return True


def _single_word_typo_match(concept: str, answer: str) -> bool:
    """Allow one small typo for a single technical concept, conservatively."""
    c = normalise(concept)
    if " " in c or len(c) < 5:
        return False
    answer_words = [w for w in normalise(answer).split() if len(w) >= 4]
    expected_forms = _inflections(c)
    for expected in expected_forms:
        if len(expected) < 5:
            continue
        for actual in answer_words:
            # First character guard reduces unrelated accidental matches.
            if expected[0] == actual[0] and _damerau_distance_at_most_one(expected, actual):
                return True
    return False


def concept_is_present(concept: str, answer: str) -> bool:
    """Match a rubric concept using transparent deterministic rules.

    Order:
    1. canonical word/phrase matching (plural and common derivations),
    2. explicitly approved semantic paraphrases,
    3. one conservative spelling/typing error for a single-word concept.

    The LLM is never used to award the official score.
    """
    actual = concept_tokens(answer)
    if _tokens_match(concept, actual):
        return True

    aliases = CONCEPT_ALIASES.get(normalise(concept), ())
    if any(_tokens_match(alias, actual) for alias in aliases):
        return True

    return _single_word_typo_match(concept, answer)


@dataclass
class RetrievalResult:
    chunk: dict
    score: float


class Retriever(Protocol):
    """Common retrieval interface used by the RAG pipeline."""

    name: str

    def retrieve(
        self,
        query: str,
        role: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        ...


class RoleAwareRetriever:
    """Role-aware TF-IDF baseline retriever."""

    name = "tfidf"

    def __init__(self, chunks: Iterable[dict]):
        self.chunks = list(chunks)
        corpus = [
            " ".join(
                [row.get("skill", ""), row.get("level", ""), row.get("text", "")]
            )
            for row in self.chunks
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        role: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        candidates = [
            i
            for i, row in enumerate(self.chunks)
            if role is None or row["role"] == role
        ]
        if not candidates:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix[candidates]).ravel()
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]),
        )
        return [
            RetrievalResult(self.chunks[i], float(score))
            for i, score in ranked[:top_k]
        ]


class EmbeddingError(RuntimeError):
    """Raised when the local embedding service cannot provide vectors."""


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...


class OllamaEmbeddingClient:
    """Small local Ollama client for dense text embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 120,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "SKILLSCOPE_EMBED_MODEL", "nomic-embed-text"
        )
        self.base_url = (
            base_url
            or os.getenv("SKILLSCOPE_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, int(batch_size))

    def _post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            raise EmbeddingError(
                "Ollama embeddings are unavailable. Start Ollama and make sure "
                f"the model '{self.model_name}' is installed."
            ) from exc

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = self._post_json(
                "/api/embed",
                {"model": self.model_name, "input": batch},
            )
            batch_vectors = payload.get("embeddings")
            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    "Ollama returned an invalid embedding response. "
                    "Check that the installed Ollama version supports /api/embed."
                )
            vectors.extend(batch_vectors)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbeddingError("Embedding vectors have an unexpected shape.")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class EmbeddingRoleAwareRetriever:
    """Role-aware dense retriever using cached local Ollama embeddings.

    Knowledge-base vectors are generated once and saved locally. At runtime,
    only the incoming query needs to be embedded. Cosine similarity is then a
    dot product because all vectors are L2-normalised.
    """

    name = "embedding"

    def __init__(
        self,
        chunks: Iterable[dict],
        embedder: EmbeddingProvider | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder or OllamaEmbeddingClient()
        self.cache_dir = cache_dir or Path(
            os.getenv(
                "SKILLSCOPE_EMBED_CACHE_DIR",
                str(ROOT / "results" / "embedding_cache"),
            )
        )
        self.use_cache = use_cache
        self.documents = [self._chunk_document(row) for row in self.chunks]
        self.fingerprint = self._fingerprint()
        self.cache_path = self.cache_dir / f"{self.fingerprint}.npz"
        self.cache_loaded = False
        self.matrix = self._load_or_build_matrix()

    @staticmethod
    def _chunk_document(row: dict) -> str:
        return " ".join(
            [row.get("skill", ""), row.get("level", ""), row.get("text", "")]
        ).strip()

    def _fingerprint(self) -> str:
        payload = {
            "model": self.embedder.model_name,
            "chunks": [
                {
                    "chunk_id": row.get("chunk_id", ""),
                    "role": row.get("role", ""),
                    "document": document,
                }
                for row, document in zip(self.chunks, self.documents)
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def _load_or_build_matrix(self) -> np.ndarray:
        if self.use_cache and self.cache_path.exists():
            try:
                with np.load(self.cache_path, allow_pickle=False) as cached:
                    matrix = np.asarray(cached["embeddings"], dtype=np.float32)
                    fingerprint = str(cached["fingerprint"].item())
                    model_name = str(cached["model_name"].item())
                if (
                    fingerprint == self.fingerprint
                    and model_name == self.embedder.model_name
                    and matrix.ndim == 2
                    and matrix.shape[0] == len(self.chunks)
                ):
                    self.cache_loaded = True
                    return matrix
            except (OSError, KeyError, ValueError):
                pass

        matrix = self.embedder.embed_texts(self.documents)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.chunks):
            raise EmbeddingError(
                "The embedding model did not return one vector per knowledge chunk."
            )

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                embeddings=matrix,
                fingerprint=np.asarray(self.fingerprint),
                model_name=np.asarray(self.embedder.model_name),
            )
        return matrix

    def retrieve(
        self,
        query: str,
        role: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        candidates = np.asarray(
            [
                i
                for i, row in enumerate(self.chunks)
                if role is None or row["role"] == role
            ],
            dtype=np.int64,
        )
        if candidates.size == 0:
            return []

        query_matrix = self.embedder.embed_texts([query])
        if query_matrix.ndim != 2 or query_matrix.shape[0] != 1:
            raise EmbeddingError("The embedding model returned an invalid query vector.")
        if query_matrix.shape[1] != self.matrix.shape[1]:
            raise EmbeddingError(
                "Query and knowledge-base embedding dimensions do not match."
            )

        scores = self.matrix @ query_matrix[0]
        ranked = sorted(
            ((int(i), float(scores[i])) for i in candidates),
            key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]),
        )
        return [
            RetrievalResult(self.chunks[i], score)
            for i, score in ranked[: max(0, int(top_k))]
        ]


def build_retriever(
    chunks: Iterable[dict],
    kind: str | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    cache_dir: Path | None = None,
) -> Retriever:
    """Build the configured retriever while retaining TF-IDF as a fallback.

    Environment variables:
      SKILLSCOPE_RETRIEVER=embedding|tfidf
      SKILLSCOPE_RETRIEVER_FALLBACK=tfidf|none
      SKILLSCOPE_EMBED_MODEL=nomic-embed-text
      SKILLSCOPE_OLLAMA_BASE_URL=http://127.0.0.1:11434
      SKILLSCOPE_EMBED_CACHE_DIR=<local cache directory>
    """
    selected = (kind or os.getenv("SKILLSCOPE_RETRIEVER", "embedding")).strip().lower()
    chunk_list = list(chunks)

    if selected in {"tfidf", "lexical"}:
        return RoleAwareRetriever(chunk_list)

    if selected in {"embedding", "dense", "nomic"}:
        try:
            return EmbeddingRoleAwareRetriever(
                chunk_list,
                embedder=embedder,
                cache_dir=cache_dir,
            )
        except EmbeddingError as exc:
            fallback = os.getenv("SKILLSCOPE_RETRIEVER_FALLBACK", "tfidf").strip().lower()
            if fallback in {"tfidf", "lexical"}:
                print(
                    "[SkillScope] Embedding retriever unavailable; "
                    f"falling back to TF-IDF. Reason: {exc}"
                )
                return RoleAwareRetriever(chunk_list)
            raise

    raise ValueError(
        "Unknown retriever. Set SKILLSCOPE_RETRIEVER to 'embedding' or 'tfidf'."
    )


def build_assessment_sets(
    questions: list[dict],
    quizzes: list[dict] | None = None,
    questions_per_set: int = 10,
    quizzes_per_set: int = 3,
) -> dict[str, list[dict]]:
    """Create five deterministic, balanced set blueprints per role.

    Questions never repeat within one set. Because the starter bank has fewer
    than 50 questions per role, base questions can appear in different sets.
    """
    grouped: dict[str, list[dict]] = {}
    for question in questions:
        grouped.setdefault(question["role"], []).append(question)

    output: dict[str, list[dict]] = {}
    for role, role_questions in sorted(grouped.items()):
        ordered = sorted(role_questions, key=lambda row: row["question_id"])
        role_quizzes = sorted(
            [row for row in (quizzes or []) if row["role"] == role],
            key=lambda row: row["question_id"],
        )
        written_count = questions_per_set - (
            quizzes_per_set if role_quizzes else 0
        )
        stride = max(1, len(ordered) // 5)
        prefix = {
            "backend_software_engineer": "B",
            "frontend_ui_developer": "U",
            "devops_sre_engineer": "D",
            "qa_test_engineer": "Q",
            "data_engineer": "E",
        }[role]
        sets = []
        for index in range(5):
            start = (index * stride) % len(ordered)
            selected = []
            for offset in range(written_count):
                row = dict(ordered[(start + offset) % len(ordered)])
                row["question_type"] = "written"
                selected.append(row)
            for offset in range(quizzes_per_set if role_quizzes else 0):
                row = dict(role_quizzes[(index + offset) % len(role_quizzes)])
                row["question_type"] = "quiz"
                selected.append(row)
            sets.append(
                {
                    "set_id": f"{prefix}{index + 1}",
                    "role": role,
                    "questions": selected,
                }
            )
        output[role] = sets
    return output


def assign_set(participant_id: str, role: str, sets: dict[str, list[dict]]) -> dict:
    digest = hashlib.sha256(f"{participant_id}:{role}".encode()).hexdigest()
    return sets[role][int(digest[:8], 16) % len(sets[role])]


def score_answer(answer: str, question: dict) -> dict:
    """Grammar-independent concept accuracy for a written answer."""
    concepts = [
        item.strip()
        for item in question["required_concepts"].split("|")
        if item.strip()
    ]
    found = [concept for concept in concepts if concept_is_present(concept, answer)]
    missing = [concept for concept in concepts if concept not in found]
    ratio = len(found) / len(concepts) if concepts else 0.0
    accuracy = round(100 * ratio, 1)
    return {
        "score": round(question.get("max_score", 10) * ratio, 1),
        "max_score": question.get("max_score", 10),
        "accuracy": accuracy,
        "competent": accuracy >= COMPETENCY_THRESHOLD,
        "matched_concepts": found,
        "missing_concepts": missing,
        "gap": (
            "none"
            if accuracy >= COMPETENCY_THRESHOLD
            else "development"
            if accuracy >= 40
            else "priority"
        ),
    }


def score_quiz(selected_index: int | None, question: dict) -> dict:
    correct = (
        selected_index is not None
        and int(selected_index) == int(question["correct_index"])
    )
    accuracy = 100.0 if correct else 0.0
    return {
        "score": question.get("max_score", 10) if correct else 0.0,
        "max_score": question.get("max_score", 10),
        "accuracy": accuracy,
        "competent": correct,
        "matched_concepts": (
            [question.get("learning_topic", "quiz concept")] if correct else []
        ),
        "missing_concepts": (
            [] if correct else [question.get("learning_topic", "quiz concept")]
        ),
        "gap": "none" if correct else "priority",
    }


def score_response(response: dict, question: dict) -> dict:
    if response.get("skipped"):
        return {
            "score": 0.0,
            "max_score": question.get("max_score", 10),
            "accuracy": 0.0,
            "competent": False,
            "matched_concepts": [],
            "missing_concepts": [
                x.strip()
                for x in question.get(
                    "required_concepts", question.get("learning_topic", "")
                ).split("|")
                if x.strip()
            ],
            "gap": "priority",
            "skipped": True,
        }
    if question.get("question_type") == "quiz":
        result = score_quiz(response.get("selected_index"), question)
    else:
        result = score_answer(str(response.get("answer", "")), question)
    result["skipped"] = False
    return result


def _invitation_secret() -> bytes:
    return os.getenv(
        "SKILLSCOPE_INVITATION_SECRET", "skillscope-local-demo-change-me"
    ).encode()


def create_invitation_token(
    participant_id: str,
    role: str,
    expires_at: int | None = None,
) -> str:
    payload = {
        "participant_id": participant_id,
        "role": role,
        "expires_at": expires_at or int(time.time()) + 7 * 86400,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).rstrip(b"=")
    signature = hmac.new(_invitation_secret(), encoded, hashlib.sha256).digest()
    return (
        f"{encoded.decode()}."
        f"{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"
    )


def verify_invitation_token(token: str) -> dict:
    try:
        encoded_text, signature_text = token.split(".", 1)
        encoded = encoded_text.encode()
        supplied = base64.urlsafe_b64decode(
            signature_text + "=" * (-len(signature_text) % 4)
        )
        expected = hmac.new(_invitation_secret(), encoded, hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("Invalid invitation signature.")
        payload = json.loads(
            base64.urlsafe_b64decode(
                encoded_text + "=" * (-len(encoded_text) % 4)
            )
        )
        if int(payload["expires_at"]) < int(time.time()):
            raise ValueError("This invitation has expired.")
        return payload
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc) in {
            "Invalid invitation signature.",
            "This invitation has expired.",
        }:
            raise
        raise ValueError("Invalid invitation token.") from exc


def load_quizzes() -> list[dict]:
    return load_jsonl(DATA_DIR / "coding_quizzes.jsonl")


def load_project_data() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        load_jsonl(DATA_DIR / "knowledge_base.jsonl"),
        load_jsonl(DATA_DIR / "assessment_questions.jsonl"),
        load_jsonl(DATA_DIR / "evaluation_cases.jsonl"),
    )
