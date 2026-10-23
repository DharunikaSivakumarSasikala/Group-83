import json
import unittest
from pathlib import Path

from skillscope_core import (
    DATA_DIR,
    build_assessment_sets,
    load_jsonl,
    load_quizzes,
    reference_chunk_ids,
    score_answer,
    score_quiz,
)


ROOT = Path(__file__).resolve().parent


class AssessmentBankIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_jsonl(DATA_DIR / "knowledge_base.jsonl")
        cls.questions = load_jsonl(DATA_DIR / "assessment_questions.jsonl")
        cls.quizzes = load_quizzes()
        cls.chunk_by_id = {row["chunk_id"]: row for row in cls.chunks}
        cls.saved_sets = json.loads((ROOT / "data" / "assessment_sets.json").read_text(encoding="utf-8"))

    def test_all_written_references_exist_and_match_role(self):
        for question in self.questions:
            ids = reference_chunk_ids(question)
            self.assertTrue(ids, question["question_id"])
            for chunk_id in ids:
                self.assertIn(chunk_id, self.chunk_by_id, question["question_id"])
                self.assertEqual(
                    self.chunk_by_id[chunk_id]["role"],
                    question["role"],
                    f"{question['question_id']} -> {chunk_id}",
                )

    def test_every_approved_reference_satisfies_its_written_rubric(self):
        """The answer shown as the approved reference must earn full rubric credit."""
        for question in self.questions:
            reference_text = " ".join(
                self.chunk_by_id[chunk_id]["text"]
                for chunk_id in reference_chunk_ids(question)
            )
            result = score_answer(reference_text, question)
            self.assertEqual(
                result["accuracy"],
                100.0,
                f"{question['question_id']} missing {result['missing_concepts']}",
            )

    def test_quiz_answer_keys_are_valid_and_exact(self):
        for quiz in self.quizzes:
            correct = quiz["correct_index"]
            self.assertGreaterEqual(correct, 0, quiz["question_id"])
            self.assertLess(correct, len(quiz["options"]), quiz["question_id"])
            self.assertEqual(score_quiz(correct, quiz)["accuracy"], 100.0)
            for index in range(len(quiz["options"])):
                if index != correct:
                    self.assertEqual(
                        score_quiz(index, quiz)["accuracy"],
                        0.0,
                        f"{quiz['question_id']} option {index}",
                    )

    def test_saved_sets_match_runtime_generation(self):
        generated = build_assessment_sets(self.questions, self.quizzes)
        self.assertEqual(self.saved_sets, generated)

    def test_every_set_has_seven_written_and_three_quizzes(self):
        self.assertEqual(sum(len(sets) for sets in self.saved_sets.values()), 25)
        for role, sets in self.saved_sets.items():
            self.assertEqual(len(sets), 5, role)
            for assessment_set in sets:
                self.assertEqual(len(assessment_set["questions"]), 10)
                written = [q for q in assessment_set["questions"] if q["question_type"] == "written"]
                quizzes = [q for q in assessment_set["questions"] if q["question_type"] == "quiz"]
                self.assertEqual(len(written), 7, assessment_set["set_id"])
                self.assertEqual(len(quizzes), 3, assessment_set["set_id"])

    def test_all_bank_items_are_represented_in_the_sets(self):
        set_written = {
            q["question_id"]
            for sets in self.saved_sets.values()
            for assessment_set in sets
            for q in assessment_set["questions"]
            if q["question_type"] == "written"
        }
        set_quizzes = {
            q["question_id"]
            for sets in self.saved_sets.values()
            for assessment_set in sets
            for q in assessment_set["questions"]
            if q["question_type"] == "quiz"
        }
        self.assertEqual(set_written, {q["question_id"] for q in self.questions})
        self.assertEqual(set_quizzes, {q["question_id"] for q in self.quizzes})

    def test_set_questions_never_cross_roles(self):
        for role, sets in self.saved_sets.items():
            for assessment_set in sets:
                self.assertEqual(assessment_set["role"], role)
                for question in assessment_set["questions"]:
                    self.assertEqual(question["role"], role, assessment_set["set_id"])


if __name__ == "__main__":
    unittest.main()
