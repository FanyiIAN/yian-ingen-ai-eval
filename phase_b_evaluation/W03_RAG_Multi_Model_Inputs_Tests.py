from __future__ import annotations

import copy
import unittest
from pathlib import Path

from W03_RAG_Multi_Model_Inputs import expand_candidate
from W03_RAG_Pipeline import build_run_inputs, load_assets
from W03_RAG_Pipeline_Tests import FakeVectorStore


ROOT = Path(__file__).resolve().parent


class MultiModelInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb, cls.eval_set, cls.config = load_assets(
            ROOT / "W03_RAG_Official_Knowledge_Base_v0.3.0.yaml",
            ROOT / "W03_RAG_Official_Blind_Eval_Set_v0.4.0.yaml",
            ROOT / "W03_RAG_Official_MultiModel_Run_Config_v0.5.0.yaml",
        )
        cls.rows = build_run_inputs(
            cls.kb,
            cls.eval_set,
            FakeVectorStore(),
            cls.config,
        )

    def test_all_candidates_preserve_messages_and_contexts(self) -> None:
        baseline_messages = [
            row["candidate_messages_sha256"] for row in self.rows
        ]
        baseline_contexts = [
            copy.deepcopy(row["retrieved_contexts"]) for row in self.rows
        ]
        for candidate in self.config["candidate_matrix"]:
            expanded, generated_config = expand_candidate(
                self.rows, self.config, candidate
            )
            self.assertEqual(
                baseline_messages,
                [row["candidate_messages_sha256"] for row in expanded],
            )
            self.assertEqual(
                baseline_contexts,
                [row["retrieved_contexts"] for row in expanded],
            )
            self.assertEqual(
                candidate["candidate_model_id"],
                generated_config["generation"]["candidate_model_id"],
            )

    def test_run_item_ids_are_unique_across_candidates(self) -> None:
        identifiers: set[str] = set()
        for candidate in self.config["candidate_matrix"]:
            expanded, _ = expand_candidate(
                self.rows, self.config, candidate
            )
            for row in expanded:
                self.assertNotIn(row["run_item_id"], identifiers)
                identifiers.add(row["run_item_id"])
        self.assertEqual(48, len(identifiers))


if __name__ == "__main__":
    unittest.main()
