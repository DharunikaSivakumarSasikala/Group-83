import unittest

from semantic_scoring import SemanticAnswerValidator, SemanticValidationError


CHUNKS = {
    "DATA_014": {
        "text": "Labels should stay carefully bounded because unbounded label values can cause harmful cardinality growth."
    },
    "DATA_016": {
        "text": "OTT ingestion pipelines should validate identifiers, titles, hierarchy, availability windows, language, artwork references and required playback metadata before publication."
    },
}

Q_LABELS = {
    "question_id": "Q_DATA_10",
    "question": "What labels should pipeline metrics avoid?",
    "required_concepts": "unbounded|cardinality",
    "reference_chunk_id": "DATA_014",
    "max_score": 10,
}

Q_METADATA = {
    "question_id": "Q_DATA_11",
    "question": "Which metadata should an OTT ingestion pipeline validate?",
    "required_concepts": "identifiers|hierarchy|availability|language",
    "reference_chunk_id": "DATA_016",
    "max_score": 10,
}


class FakeGenerator:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, system_prompt, user_prompt):
        return self.payload


class BrokenGenerator:
    def generate_json(self, system_prompt, user_prompt):
        raise RuntimeError("offline")


class SemanticScoringTests(unittest.TestCase):
    def test_correct_paraphrase_can_receive_full_credit(self):
        payload = {
            "evaluations": [{
                "question_id": "Q_DATA_11",
                "concepts": [
                    {"concept": "identifiers", "status": "demonstrated", "reason": "content identifiers"},
                    {"concept": "hierarchy", "status": "demonstrated", "reason": "how metadata is organised"},
                    {"concept": "availability", "status": "demonstrated", "reason": "when content is available"},
                    {"concept": "language", "status": "demonstrated", "reason": "language information"},
                ],
                "overall_consistency": "consistent",
                "confidence": "high",
            }]
        }
        validator = SemanticAnswerValidator(CHUNKS, FakeGenerator(payload))
        result = validator.validate_batch([(Q_METADATA, "paraphrased answer")])["Q_DATA_11"]
        self.assertEqual(result["accuracy"], 100.0)
        self.assertTrue(result["competent"])
        self.assertEqual(result["missing_concepts"], [])

    def test_keyword_contradiction_does_not_receive_full_credit(self):
        payload = {
            "evaluations": [{
                "question_id": "Q_DATA_10",
                "concepts": [
                    {"concept": "unbounded", "status": "demonstrated", "reason": "unbounded labels are identified"},
                    {"concept": "cardinality", "status": "contradicted", "reason": "answer says high cardinality improves monitoring"},
                ],
                "overall_consistency": "contradicted",
                "confidence": "high",
            }]
        }
        validator = SemanticAnswerValidator(CHUNKS, FakeGenerator(payload))
        result = validator.validate_batch([(Q_LABELS, "Unbounded labels create high cardinality and make monitoring efficient.")])["Q_DATA_10"]
        self.assertEqual(result["accuracy"], 50.0)
        self.assertFalse(result["competent"])
        self.assertEqual(result["contradicted_concepts"], ["cardinality"])

    def test_missing_concepts_are_not_inferred(self):
        payload = {
            "evaluations": [{
                "question_id": "Q_DATA_11",
                "concepts": [
                    {"concept": "identifiers", "status": "demonstrated", "reason": "identifiers stated"},
                    {"concept": "hierarchy", "status": "absent", "reason": "not expressed"},
                    {"concept": "availability", "status": "absent", "reason": "not expressed"},
                    {"concept": "language", "status": "absent", "reason": "not expressed"},
                ],
                "overall_consistency": "consistent",
                "confidence": "high",
            }]
        }
        validator = SemanticAnswerValidator(CHUNKS, FakeGenerator(payload))
        result = validator.validate_batch([(Q_METADATA, "Check identifiers.")])["Q_DATA_11"]
        self.assertEqual(result["accuracy"], 25.0)
        self.assertEqual(result["missing_concepts"], ["hierarchy", "availability", "language"])

    def test_safe_fallback_preserves_assessment_if_validator_is_unavailable(self):
        validator = SemanticAnswerValidator(CHUNKS, BrokenGenerator())
        result = validator.validate_batch_with_fallback(
            [(Q_LABELS, "Avoid unbounded labels because they create high cardinality.")]
        )["Q_DATA_10"]
        self.assertEqual(result["scoring_mode"], "deterministic_fallback")
        self.assertTrue(result["human_review_required"])
        self.assertIn("offline", result["semantic_validation_error"])

    def test_invalid_concept_structure_is_rejected(self):
        payload = {
            "evaluations": [{
                "question_id": "Q_DATA_10",
                "concepts": [
                    {"concept": "unbounded", "status": "demonstrated", "reason": "present"},
                ],
                "overall_consistency": "consistent",
                "confidence": "high",
            }]
        }
        validator = SemanticAnswerValidator(CHUNKS, FakeGenerator(payload))
        with self.assertRaises(SemanticValidationError):
            validator.validate_batch([(Q_LABELS, "answer")])


if __name__ == "__main__":
    unittest.main()
