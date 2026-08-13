from __future__ import annotations

import unittest

from W05_RAG_Coverage_Scoring import normalize_coverage


ITEM = {
    "required_points": [
        {"point_id": "P1", "weight": 2, "criterion": "registered"}
    ],
    "forbidden_points": [],
}


class CoverageNormalizationTests(unittest.TestCase):
    def test_unregistered_extra_is_discarded_and_audited(self) -> None:
        result = normalize_coverage(
            {
                "point_scores": [
                    {"point_id": "P1", "score": 0.5, "evidence": "partial"},
                    {"point_id": "P2", "score": 0, "evidence": ""},
                ],
                "forbidden_point_violations": [],
                "rationale": "ok",
            },
            ITEM,
        )
        self.assertEqual(result["required_point_coverage"], 0.5)
        self.assertEqual(result["ignored_unknown_point_ids"], ["P2"])

    def test_missing_registered_point_is_never_imputed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing point scores"):
            normalize_coverage(
                {
                    "point_scores": [
                        {"point_id": "P2", "score": 1, "evidence": "extra"}
                    ],
                    "forbidden_point_violations": [],
                },
                ITEM,
            )


if __name__ == "__main__":
    unittest.main()
