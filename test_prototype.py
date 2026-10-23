import unittest

from skillscope_core import RoleAwareRetriever, assign_set, build_assessment_sets, load_project_data, load_quizzes, score_answer


class SkillScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.questions, cls.cases = load_project_data()
        cls.quizzes = load_quizzes()
        cls.sets = build_assessment_sets(cls.questions, cls.quizzes)

    def test_dataset_counts(self):
        self.assertEqual((len(self.chunks), len(self.questions), len(self.cases)), (120, 83, 249))

    def test_twenty_five_balanced_sets(self):
        self.assertEqual(sum(map(len, self.sets.values())), 25)
        for role_sets in self.sets.values():
            self.assertTrue(all(len(item["questions"]) == 10 for item in role_sets))
            for item in role_sets:
                ids = [q["question_id"] for q in item["questions"]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(sum(q["question_type"] == "quiz" for q in item["questions"]), 3)
                self.assertEqual(sum(q["question_type"] == "written" for q in item["questions"]), 7)

    def test_assignment_is_stable(self):
        first = assign_set("EMP-42", "qa_test_engineer", self.sets)["set_id"]
        second = assign_set("EMP-42", "qa_test_engineer", self.sets)["set_id"]
        self.assertEqual(first, second)

    def test_role_filter(self):
        retriever = RoleAwareRetriever(self.chunks)
        results = retriever.retrieve("How should an HLS manifest be tested?", "qa_test_engineer", 5)
        self.assertTrue(results)
        self.assertTrue(all(item.chunk["role"] == "qa_test_engineer" for item in results))

    def test_transparent_scoring(self):
        question = {"required_concepts": "caller|access|object", "max_score": 10}
        result = score_answer("Verify the caller has access to the object.", question)
        self.assertEqual(result["score"], 10.0)
        self.assertEqual(result["accuracy"], 100.0)
        self.assertEqual(result["gap"], "none")


if __name__ == "__main__":
    unittest.main()
