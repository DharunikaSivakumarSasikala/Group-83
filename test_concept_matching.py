import unittest

from skillscope_core import concept_is_present, score_answer


class DeterministicConceptMatchingTests(unittest.TestCase):
    def test_message_keys_matches_key(self):
        self.assertTrue(
            concept_is_present(
                "key",
                "We define the event schema, message keys, ordering and replay behaviour.",
            )
        )

    def test_playback_paraphrases_receive_partial_credit(self):
        question = {
            "required_concepts": "duplicate|late|malformed",
            "max_score": 10,
        }
        result = score_answer(
            "Detect events arriving more than once and events arriving later than expected.",
            question,
        )
        self.assertEqual(result["accuracy"], 66.7)
        self.assertEqual(result["matched_concepts"], ["duplicate", "late"])
        self.assertEqual(result["missing_concepts"], ["malformed"])
        self.assertFalse(result["competent"])

    def test_missing_concept_is_not_inferred(self):
        self.assertFalse(
            concept_is_present(
                "malformed",
                "Detect events arriving more than once and events arriving later than expected.",
            )
        )

    def test_metadata_aliases_are_explicit(self):
        question = {
            "required_concepts": "counts|rejected records|field mappings|samples",
            "max_score": 10,
        }
        result = score_answer(
            "Compare row counts, failed records, column mappings and representative rows.",
            question,
        )
        self.assertEqual(result["accuracy"], 100.0)
        self.assertTrue(result["competent"])

    def test_unrelated_semantic_guess_does_not_get_credit(self):
        self.assertFalse(concept_is_present("duplicate", "The event has a quality problem."))

    def test_plural_and_derivational_forms(self):
        self.assertTrue(concept_is_present("range", "Values must stay within acceptable ranges."))
        self.assertTrue(concept_is_present("failure", "The pipeline fails repeatedly."))
        self.assertTrue(concept_is_present("validate", "The payload is validated before use."))
        self.assertTrue(concept_is_present("cache", "Instruction order improves build caching."))

    def test_latency_paraphrase_is_controlled(self):
        self.assertTrue(
            concept_is_present(
                "latency",
                "Notify the team when the pipeline takes much longer than expected.",
            )
        )

    def test_one_transposition_typo_is_tolerated(self):
        # "retired" is one adjacent transposition away from the valid inflection
        # "retried". This protects a genuine spelling slip without using an LLM.
        self.assertTrue(
            concept_is_present(
                "retry",
                "If the task needs to be retired, running it again must stay safe.",
            )
        )

    def test_typo_tolerance_does_not_become_semantic_guessing(self):
        self.assertFalse(concept_is_present("latency", "The pipeline has a problem."))
        self.assertFalse(concept_is_present("duplicate", "The result is inaccurate."))


if __name__ == "__main__":
    unittest.main()
