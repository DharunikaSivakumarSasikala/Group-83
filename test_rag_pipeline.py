import unittest

from rag_pipeline import GenerationError, RAGEvaluator, build_augmented_prompt, validate_generation
from skillscope_core import RoleAwareRetriever, load_project_data


class FakeGenerator:
    def __init__(self, payload):
        self.payload = payload
        self.user_prompt = ""

    def generate_json(self, system_prompt, user_prompt):
        self.user_prompt = user_prompt
        return self.payload


class BrokenGenerator:
    def generate_json(self, system_prompt, user_prompt):
        raise GenerationError("offline")


class RAGPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.questions, _ = load_project_data()
        cls.question = cls.questions[0]
        cls.retriever = RoleAwareRetriever(cls.chunks)

    def test_prompt_contains_all_rag_inputs(self):
        prompt = build_augmented_prompt(self.question, "Example employee answer", [self.chunks[0]])
        for label in ["ASSESSMENT QUESTION", "EMPLOYEE ANSWER", "RUBRIC", "EVIDENCE"]:
            self.assertIn(label, prompt)
        self.assertIn(self.chunks[0]["chunk_id"], prompt)

    def test_end_to_end_rag_with_valid_citation(self):
        expected_id = self.question["reference_chunk_id"]
        generator = FakeGenerator(
            {
                "score": 7,
                "strengths": ["Explains the main purpose"],
                "missing_concepts": ["outputs"],
                "improvement_areas": [
                    {
                        "area": "API contracts",
                        "priority": "medium",
                        "reason": "Output contracts were omitted",
                        "recommended_activity": "Review response schemas",
                    }
                ],
                "evidence_chunk_ids": [expected_id],
                "confidence": "medium",
                "needs_human_review": True,
                "rationale": "The answer covers most rubric concepts.",
            }
        )
        evaluator = RAGEvaluator(self.retriever, generator=generator, top_k=40)
        result = evaluator.evaluate(self.question, "It describes operations and inputs.")
        self.assertEqual(result["mode"], "rag")
        self.assertIn(expected_id, result["evidence_chunk_ids"])
        self.assertTrue(result["human_review_required"])
        self.assertIn("<employee_answer>", generator.user_prompt)

    def test_unretrieved_citation_is_rejected(self):
        with self.assertRaises(GenerationError):
            validate_generation({"score": 5, "evidence_chunk_ids": ["FAKE_999"]}, {"BACK_001"}, 10)

    def test_score_is_clamped(self):
        result = validate_generation({"score": 99, "evidence_chunk_ids": []}, {"BACK_001"}, 10)
        self.assertEqual(result["score"], 10.0)

    def test_safe_fallback_is_labelled_not_rag(self):
        evaluator = RAGEvaluator(self.retriever, generator=BrokenGenerator())
        result = evaluator.evaluate_with_safe_fallback(self.question, "operations inputs")
        self.assertEqual(result["mode"], "retrieval_only_fallback")
        self.assertEqual(result["confidence"], "low")
        self.assertTrue(result["human_review_required"])


if __name__ == "__main__":
    unittest.main()
