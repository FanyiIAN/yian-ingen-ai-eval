from __future__ import annotations

import unittest
from pathlib import Path

from W03_RAG_Retrieval_Ablation import (
    RERANKER_MODEL_ID,
    configure_variant,
    registered_variants,
)
from W03_RAG_Pipeline import load_assets


ROOT = Path(__file__).resolve().parent


class RetrievalAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, _, cls.config = load_assets(
            ROOT / "W03_RAG_Official_Knowledge_Base_v0.3.0.yaml",
            ROOT / "W03_RAG_Official_Eval_Set_v0.3.0.yaml",
            ROOT / "W03_RAG_Official_Run_Config_v0.3.0.yaml",
        )

    def test_registered_matrix_has_18_unique_variants(self) -> None:
        variants = registered_variants()
        self.assertEqual(18, len(variants))
        keys = {
            (
                row["chunk_size_tokens"],
                row["top_k"],
                row["reranker_enabled"],
            )
            for row in variants
        }
        self.assertEqual(18, len(keys))

    def test_variant_changes_only_registered_retrieval_fields(self) -> None:
        variant = {
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 64,
            "top_k": 3,
            "reranker_enabled": True,
        }
        configured = configure_variant(
            self.config, variant, Path("/workspace/models/reranker")
        )
        self.assertEqual(
            512,
            configured["retrieval"]["text_splitter"]["chunk_size_tokens"],
        )
        self.assertEqual(3, configured["retrieval"]["retriever"]["top_k"])
        self.assertTrue(configured["retrieval"]["reranker"]["enabled"])
        self.assertEqual(
            RERANKER_MODEL_ID,
            configured["retrieval"]["reranker"]["model_id"],
        )
        self.assertEqual(
            self.config["generation"], configured["generation"]
        )


if __name__ == "__main__":
    unittest.main()
