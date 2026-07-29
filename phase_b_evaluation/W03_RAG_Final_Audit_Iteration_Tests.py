from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from W03_RAG_Final_Audit_Iteration import (
    OFFICIAL_RAG_SYSTEM_PROMPT_V033,
    OFFICIAL_RAG_SYSTEM_PROMPT_V034,
    build_final_audit_iteration,
    canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parent


class FinalAuditIterationTests(unittest.TestCase):
    def setUp(self) -> None:
        with (ROOT / "W03_RAG_Official_Run_Config_v0.3.4.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            self.config = yaml.safe_load(handle)

    def parent_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(12):
            messages = [
                {
                    "role": "system",
                    "content": OFFICIAL_RAG_SYSTEM_PROMPT_V033,
                },
                {"role": "user", "content": "compound question"},
            ]
            rows.append(
                {
                    "run_item_id": f"EVAL-{index:02d}::rag::prompt-v0.3.3",
                    "eval_id": f"EVAL-{index:02d}",
                    "condition": "rag",
                    "candidate_model_id": self.config["generation"][
                        "candidate_model_id"
                    ],
                    "candidate_model_revision": self.config["generation"][
                        "candidate_model_revision"
                    ],
                    "random_seed": 42,
                    "candidate_messages": messages,
                    "candidate_messages_sha256": canonical_json_sha256(
                        messages
                    ),
                    "retrieved_contexts": [
                        {"chunk_id": "C1", "content": "governed fact"}
                    ],
                    "parent_run_item_id": (
                        f"EVAL-{index:02d}::rag::prompt-v0.3.2"
                    ),
                    "prior_lineage": {},
                }
            )
        return rows

    def test_only_prompt_and_lineage_change(self) -> None:
        parent = self.parent_rows()
        frozen_contexts = [
            copy.deepcopy(row["retrieved_contexts"]) for row in parent
        ]
        rows, validation = build_final_audit_iteration(parent, self.config)
        self.assertEqual(12, len(rows))
        self.assertEqual({"rag": 12}, validation["condition_counts"])
        for index, row in enumerate(rows):
            self.assertEqual(frozen_contexts[index], row["retrieved_contexts"])
            self.assertEqual(
                OFFICIAL_RAG_SYSTEM_PROMPT_V034,
                row["candidate_messages"][0]["content"],
            )
            self.assertTrue(row["run_item_id"].endswith("::prompt-v0.3.4"))
            self.assertTrue(
                row["parent_run_item_id"].endswith("::prompt-v0.3.3")
            )
            self.assertIn(
                "Final epistemic audit",
                row["candidate_messages"][0]["content"],
            )

    def test_rejects_modified_parent_prompt(self) -> None:
        parent = self.parent_rows()
        parent[0]["candidate_messages"][0]["content"] = "unregistered prompt"
        with self.assertRaisesRegex(ValueError, "unexpected v0.3.3 prompt"):
            build_final_audit_iteration(parent, self.config)


if __name__ == "__main__":
    unittest.main()

