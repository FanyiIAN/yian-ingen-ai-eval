import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from W04_Freeze_Robustness_Inputs import (
    read_jsonl,
    request_ids_sha256,
    validate_and_apply,
)


HERE = Path(__file__).resolve().parent
PRIVATE_DRAFT = (
    HERE.parent.parent
    / "private"
    / "phase_b_evaluation"
    / "runs"
    / "w04-robustness-input-draft-v010"
    / "W04_Robustness_Inputs_DRAFT.jsonl"
)
REVIEW = HERE / "W04_Robustness_Semantic_Equivalence_Review_v0.1.0.yaml"


class FreezeRobustnessInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(PRIVATE_DRAFT)
        cls.review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))

    def test_request_id_hash_is_order_independent(self):
        ids = ["b", "a"]
        self.assertEqual(request_ids_sha256(ids), request_ids_sha256(ids[::-1]))

    def test_anchored_review_freezes_all_variants(self):
        frozen = validate_and_apply(self.rows, self.review, PRIVATE_DRAFT)
        reviewed = [
            row
            for row in frozen
            if row.get("semantic_equivalence_review") == "approved_ai_assisted"
        ]
        self.assertEqual(len(reviewed), 105)
        self.assertTrue(all(row.get("semantic_equivalence_review_id") for row in reviewed))

    def test_hash_mismatch_is_rejected(self):
        review = copy.deepcopy(self.review)
        review["reviewed_artifact"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not anchored"):
            validate_and_apply(self.rows, review, PRIVATE_DRAFT)

    def test_unresolved_exception_is_rejected(self):
        review = copy.deepcopy(self.review)
        review["exceptions"] = ["w04-semantic::FARI-001::tone_shift"]
        with self.assertRaisesRegex(ValueError, "unresolved exceptions"):
            validate_and_apply(self.rows, review, PRIVATE_DRAFT)


if __name__ == "__main__":
    unittest.main()

