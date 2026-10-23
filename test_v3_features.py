import time
import unittest

from evaluate_rag_quality import calculate_metrics
from skillscope_core import (
    create_invitation_token, load_quizzes, score_answer, score_response,
    verify_invitation_token,
)


class SkillScopeV3Tests(unittest.TestCase):
    def test_word_variants_are_matched_without_grammar_marking(self):
        question = {"required_concepts": "identify|object", "max_score": 10}
        result = score_answer("The service identifies and identified the object.", question)
        self.assertEqual(result["accuracy"], 100.0)
        self.assertTrue(result["competent"])

    def test_eighty_percent_threshold(self):
        question = {"required_concepts": "one|two|three|four|five", "max_score": 10}
        result = score_answer("one two three four", question)
        self.assertEqual(result["accuracy"], 80.0)
        self.assertTrue(result["competent"])

    def test_skip_scores_zero(self):
        question = {"required_concepts": "actual|expected", "max_score": 10, "question_type": "written"}
        result = score_response({"skipped": True}, question)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertTrue(result["skipped"])

    def test_quiz_exact_scoring(self):
        quiz = dict(load_quizzes()[0]); quiz["question_type"] = "quiz"
        correct = score_response({"selected_index": quiz["correct_index"]}, quiz)
        wrong = score_response({"selected_index": (quiz["correct_index"] + 1) % 4}, quiz)
        self.assertEqual((correct["accuracy"], wrong["accuracy"]), (100.0, 0.0))

    def test_signed_invitation_round_trip_and_tamper_rejection(self):
        token = create_invitation_token("EMP-42", "qa_test_engineer")
        self.assertEqual(verify_invitation_token(token)["participant_id"], "EMP-42")
        payload, signature = token.split(".", 1)
        tampered = payload + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
        with self.assertRaises(ValueError):
            verify_invitation_token(tampered)

    def test_expired_invitation_rejected(self):
        token = create_invitation_token("EMP-42", "qa_test_engineer", int(time.time()) - 1)
        with self.assertRaisesRegex(ValueError, "expired"):
            verify_invitation_token(token)

    def test_hallucination_metrics(self):
        metrics = calculate_metrics([{
            "human_score": 80, "model_score": 70,
            "cited_chunk_ids": ["A", "X"], "allowed_chunk_ids": ["A", "B"],
            "expected_citation_ids": ["A"], "claim_support": [True, False],
        }])
        self.assertEqual(metrics["score_mae"], 10.0)
        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["hallucination_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
