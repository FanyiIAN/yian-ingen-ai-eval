"""Contract tests for the expanded Week 3 public RAG assets."""

from __future__ import annotations

import re
import unittest
from collections import Counter
from pathlib import Path

import yaml

from W03_RAG_Pipeline import validate_assets


ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "W03_RAG_Expanded_Knowledge_Base_v0.6.0.yaml"
EVAL_PATH = ROOT / "W03_RAG_Expanded_Eval_Set_v0.6.0.yaml"
CONFIG_PATH = ROOT / "W03_RAG_Expanded_Run_Config_v0.6.0.yaml"
ITERATION_CONFIG_PATH = ROOT / "W03_RAG_Expanded_Run_Config_v0.6.1.yaml"
MULTI_MODEL_CONFIG_PATH = (
    ROOT / "W03_RAG_Expanded_MultiModel_Run_Config_v0.6.2.yaml"
)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class ExpandedRAGAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = load(KB_PATH)
        cls.eval_set = load(EVAL_PATH)
        cls.config = load(CONFIG_PATH)
        cls.iteration_config = load(ITERATION_CONFIG_PATH)
        cls.multi_model_config = load(MULTI_MODEL_CONFIG_PATH)

    def test_pipeline_contract_validates(self) -> None:
        summary = validate_assets(self.kb, self.eval_set, self.config)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["evaluation_items"], 40)

    def test_collection_is_materially_expanded_and_balanced(self) -> None:
        sections = [
            (document["platform"], section)
            for document in self.kb["documents"]
            for section in document["sections"]
        ]
        counts = Counter(platform for platform, _ in sections)
        self.assertGreaterEqual(len(sections), 200)
        self.assertGreaterEqual(counts["Fari"], 100)
        self.assertGreaterEqual(counts["Senpai"], 100)
        self.assertEqual(self.kb["atomic_section_count"], len(sections))

    def test_atomic_units_are_traceable_and_compact(self) -> None:
        for document in self.kb["documents"]:
            source = document["source"]
            self.assertRegex(source["snapshot_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(source["domain"], "www.ingendynamics.com")
            self.assertEqual(source["accessed_at"], "2026-08-04")
            for section in document["sections"]:
                self.assertTrue(section["source_fragment"])
                self.assertGreater(section["source_record_index"], 0)
                self.assertTrue(section["atomic_unit_type"])
                self.assertTrue(section["claim_status"])
                self.assertLessEqual(len(section["content"].split()), 160)
                self.assertIsNone(re.search(r"<[^>]+>", section["content"]))

    def test_question_set_is_40_balanced_and_diverse(self) -> None:
        items = self.eval_set["items"]
        self.assertEqual(len(items), 40)
        self.assertEqual(Counter(row["platform"] for row in items), {"Fari": 20, "Senpai": 20})
        self.assertEqual(len({row["question"] for row in items}), 40)
        self.assertGreaterEqual(sum(row["difficulty"] == "hard" for row in items), 15)
        self.assertGreaterEqual(len({row["question_type"] for row in items}), 15)
        self.assertGreaterEqual(sum(len(row["evidence_fact_ids"]) >= 3 for row in items), 15)

    def test_all_scoring_points_are_evidence_linked(self) -> None:
        facts = {
            fact["fact_id"]
            for document in self.kb["documents"]
            for fact in document["supported_facts"]
        }
        for row in self.eval_set["items"]:
            evidence = set(row["evidence_fact_ids"])
            self.assertTrue(evidence)
            self.assertTrue(evidence <= facts)
            self.assertTrue(row["required_points"])
            self.assertTrue(row["forbidden_points"])
            for point in row["required_points"]:
                self.assertTrue(set(point["evidence_fact_ids"]) <= evidence)

    def test_expanded_retrieval_is_not_the_old_smoke_configuration(self) -> None:
        retriever = self.config["retrieval"]["retriever"]
        reranker = self.config["retrieval"]["reranker"]
        self.assertEqual(retriever["fetch_k"], 24)
        self.assertEqual(retriever["top_k"], 6)
        self.assertTrue(reranker["enabled"])
        self.assertEqual(reranker["model_id"], "BAAI/bge-reranker-v2-m3")

    def test_registered_retrieval_iteration_changes_only_search_breadth(self) -> None:
        summary = validate_assets(
            self.kb, self.eval_set, self.iteration_config
        )
        self.assertEqual(summary["status"], "ok")
        baseline = self.config["retrieval"]
        iteration = self.iteration_config["retrieval"]
        self.assertEqual(baseline["embedding"], iteration["embedding"])
        self.assertEqual(baseline["text_splitter"], iteration["text_splitter"])
        self.assertEqual(iteration["retriever"]["fetch_k"], 32)
        self.assertEqual(iteration["retriever"]["top_k"], 8)

    def test_three_model_contract_registers_240_paired_rows(self) -> None:
        summary = validate_assets(
            self.kb, self.eval_set, self.multi_model_config
        )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(
            self.multi_model_config["comparison"]["total_candidate_rows"],
            240,
        )
        candidates = self.multi_model_config["candidate_matrix"]
        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            self.multi_model_config["retrieval"]["retriever"]["top_k"],
            10,
        )
        self.assertEqual(
            {candidate["candidate_model_key"] for candidate in candidates},
            {
                "flan_t5_base",
                "mistral_7b_instruct_v0_2",
                "llama31_8b_instruct",
            },
        )


if __name__ == "__main__":
    unittest.main()
