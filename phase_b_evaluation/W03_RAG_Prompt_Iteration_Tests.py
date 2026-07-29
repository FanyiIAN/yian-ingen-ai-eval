import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "w03_rag_prompt_iteration",
    ROOT / "W03_RAG_Prompt_Iteration.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PromptIterationTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / "W03_RAG_Official_Run_Config_v0.3.1.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            self.config = yaml.safe_load(handle)

    def parent_rows(self):
        rows = []
        for index in range(12):
            eval_id = f"EVAL-{index:02d}"
            for condition in ("base", "rag"):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            MODULE.OFFICIAL_RAG_SYSTEM_PROMPT_V030
                            if condition == "rag"
                            else "base prompt"
                        ),
                    },
                    {"role": "user", "content": "question"},
                ]
                rows.append(
                    {
                        "run_item_id": f"{eval_id}::{condition}",
                        "eval_id": eval_id,
                        "condition": condition,
                        "candidate_model_id": self.config["generation"][
                            "candidate_model_id"
                        ],
                        "candidate_model_revision": self.config["generation"][
                            "candidate_model_revision"
                        ],
                        "random_seed": 42,
                        "candidate_messages": messages,
                        "candidate_messages_sha256": MODULE.canonical_json_sha256(
                            messages
                        ),
                        "retrieved_contexts": (
                            [{"chunk_id": "C1", "content": "fact"}]
                            if condition == "rag"
                            else []
                        ),
                    }
                )
        return rows

    def test_changes_only_rag_prompt_and_preserves_context(self):
        parent = self.parent_rows()
        rows, validation = MODULE.build_prompt_iteration(parent, self.config)
        self.assertEqual(12, len(rows))
        self.assertEqual({"rag": 12}, validation["condition_counts"])
        for row in rows:
            self.assertEqual("rag", row["condition"])
            self.assertEqual([{"chunk_id": "C1", "content": "fact"}], row["retrieved_contexts"])
            self.assertIn("Preserve epistemic polarity", row["candidate_messages"][0]["content"])
            self.assertIn("::prompt-v0.3.1", row["run_item_id"])

    def test_rejects_unexpected_parent_prompt(self):
        parent = self.parent_rows()
        rag_row = next(row for row in parent if row["condition"] == "rag")
        rag_row["candidate_messages"][0]["content"] = "changed parent prompt"
        with self.assertRaisesRegex(ValueError, "unexpected v0.3.0 prompt"):
            MODULE.build_prompt_iteration(parent, self.config)


if __name__ == "__main__":
    unittest.main()
