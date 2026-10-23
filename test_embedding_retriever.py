import tempfile
import unittest
from pathlib import Path

import numpy as np

from skillscope_core import (
    EmbeddingRoleAwareRetriever,
    RoleAwareRetriever,
    build_retriever,
)


class FakeEmbedder:
    model_name = "fake-test-model"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float("permission" in lower or "authorization" in lower),
                    float("keyboard" in lower or "focus" in lower),
                    float("pipeline" in lower or "dag" in lower),
                ]
            )
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


CHUNKS = [
    {
        "chunk_id": "BACK_001",
        "role": "backend_software_engineer",
        "skill": "API authorization",
        "level": "intermediate",
        "text": "Check authorization before allowing access to an object.",
    },
    {
        "chunk_id": "BACK_002",
        "role": "backend_software_engineer",
        "skill": "General API",
        "level": "intermediate",
        "text": "Describe normal HTTP request and response behaviour.",
    },
    {
        "chunk_id": "FRON_001",
        "role": "frontend_ui_developer",
        "skill": "Keyboard access",
        "level": "intermediate",
        "text": "Interactive controls need keyboard access and visible focus.",
    },
    {
        "chunk_id": "DATA_001",
        "role": "data_engineer",
        "skill": "DAGs",
        "level": "intermediate",
        "text": "A DAG represents a scheduled data pipeline.",
    },
]


class EmbeddingRetrieverTests(unittest.TestCase):
    def test_embedding_retrieval_respects_role_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            retriever = EmbeddingRoleAwareRetriever(
                CHUNKS,
                embedder=FakeEmbedder(),
                cache_dir=Path(tmp),
            )
            results = retriever.retrieve(
                "How do we check user permission for a record?",
                role="backend_software_engineer",
                top_k=2,
            )
            self.assertEqual(results[0].chunk["chunk_id"], "BACK_001")
            self.assertTrue(
                all(
                    item.chunk["role"] == "backend_software_engineer"
                    for item in results
                )
            )

    def test_document_embeddings_are_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            first_embedder = FakeEmbedder()
            first = EmbeddingRoleAwareRetriever(
                CHUNKS,
                embedder=first_embedder,
                cache_dir=cache_dir,
            )
            self.assertFalse(first.cache_loaded)
            self.assertEqual(len(first_embedder.calls), 1)

            second_embedder = FakeEmbedder()
            second = EmbeddingRoleAwareRetriever(
                CHUNKS,
                embedder=second_embedder,
                cache_dir=cache_dir,
            )
            self.assertTrue(second.cache_loaded)
            self.assertEqual(second_embedder.calls, [])

    def test_factory_can_force_tfidf_baseline(self):
        retriever = build_retriever(CHUNKS, kind="tfidf")
        self.assertIsInstance(retriever, RoleAwareRetriever)
        self.assertEqual(retriever.name, "tfidf")

    def test_factory_can_build_embedding_retriever(self):
        with tempfile.TemporaryDirectory() as tmp:
            retriever = build_retriever(
                CHUNKS,
                kind="embedding",
                embedder=FakeEmbedder(),
                cache_dir=Path(tmp),
            )
            self.assertIsInstance(retriever, EmbeddingRoleAwareRetriever)
            self.assertEqual(retriever.name, "embedding")


if __name__ == "__main__":
    unittest.main()
