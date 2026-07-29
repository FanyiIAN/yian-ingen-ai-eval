"""CPU-only contract tests for the Week 3 RAG smoke pipeline."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import W03_RAG_Evaluation as evaluation
import W03_RAG_Generation as generation
import W03_RAG_Pipeline as pipeline


class FakeVectorStore:
    def __init__(self) -> None:
        self._collection = SimpleNamespace(name="fake_test_collection")

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
        filter: dict[str, str] | None,
    ) -> list[tuple[SimpleNamespace, float]]:
        platform = (filter or {}).get("platform", "Fari")
        document_id = (
            "FARI-SMOKE-001" if platform == "Fari" else "SENPAI-SMOKE-001"
        )
        document = SimpleNamespace(
            page_content=f"Fixture context for {query}",
            metadata={
                "chunk_id": f"{document_id}::chunk-001",
                "document_id": document_id,
                "document_version": "1.0.0",
                "title": "Fixture",
                "chunk_content_sha256": pipeline.sha256_text(
                    f"Fixture context for {query}"
                ),
                "start_index": 0,
            },
        )
        return [(document, 0.9)][:k]


class FakeRerankVectorStore:
    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
        filter: dict[str, str] | None,
    ) -> list[tuple[SimpleNamespace, float]]:
        documents = [
            SimpleNamespace(
                page_content="Dense-first context",
                metadata={
                    "chunk_id": "A",
                    "document_id": "DOC-A",
                    "document_version": "1",
                    "chunk_content_sha256": "a" * 64,
                },
            ),
            SimpleNamespace(
                page_content="Cross-encoder-first context",
                metadata={
                    "chunk_id": "B",
                    "document_id": "DOC-B",
                    "document_version": "1",
                    "chunk_content_sha256": "b" * 64,
                },
            ),
        ]
        return list(zip(documents, (0.95, 0.80), strict=True))[:k]


class FakeReranker:
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return [0.1, 0.9]


class RAGPipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb, cls.eval_set, cls.config = pipeline.load_assets()
        cls.official_kb, cls.official_eval_set, cls.official_config = (
            pipeline.load_assets(
                pipeline.SCRIPT_DIR
                / "W03_RAG_Official_Knowledge_Base_v0.3.0.yaml",
                pipeline.SCRIPT_DIR / "W03_RAG_Official_Eval_Set_v0.3.0.yaml",
                pipeline.SCRIPT_DIR / "W03_RAG_Official_Run_Config_v0.3.0.yaml",
            )
        )
        _, cls.blind_eval_set, cls.blind_config = pipeline.load_assets(
            pipeline.SCRIPT_DIR
            / "W03_RAG_Official_Knowledge_Base_v0.3.0.yaml",
            pipeline.SCRIPT_DIR
            / "W03_RAG_Official_Blind_Eval_Set_v0.4.0.yaml",
            pipeline.SCRIPT_DIR
            / "W03_RAG_Official_Run_Config_v0.4.1.yaml",
        )

    def test_assets_validate_as_smoke_only(self) -> None:
        result = pipeline.validate_assets(self.kb, self.eval_set, self.config)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["documents"], 4)
        self.assertEqual(result["evaluation_items"], 6)
        self.assertEqual(result["expected_generation_rows"], 12)
        self.assertTrue(result["warnings"])

    def test_required_point_must_reference_item_evidence(self) -> None:
        broken = copy.deepcopy(self.eval_set)
        broken["items"][0]["required_points"][0]["evidence_fact_ids"] = [
            "FARI-SMOKE-002-F1"
        ]
        with self.assertRaisesRegex(ValueError, "point evidence is not item evidence"):
            pipeline.validate_assets(self.kb, broken, self.config)

    def test_official_assets_validate_with_source_governance(self) -> None:
        result = pipeline.validate_assets(
            self.official_kb,
            self.official_eval_set,
            self.official_config,
        )
        self.assertEqual(result["data_origin"], "official_public_curated")
        self.assertEqual(result["documents"], 4)
        self.assertEqual(result["sections"], 16)
        self.assertEqual(result["evaluation_items"], 12)
        source_documents = pipeline.build_source_documents(self.official_kb)
        self.assertEqual(len(source_documents), 16)
        required = {
            "owner_type",
            "access_scope",
            "source_url",
            "source_domain",
            "accessed_at",
            "claim_status",
            "section_path",
            "parent_chunk_id",
        }
        self.assertTrue(required.issubset(source_documents[0].metadata))

    def test_official_metadata_gate_is_strict(self) -> None:
        item = self.official_eval_set["items"][0]
        where = pipeline.metadata_filter_for_item(
            item, self.official_config
        )
        self.assertIn("$and", where)
        rendered = pipeline.canonical_json_sha256(where)
        self.assertEqual(len(rendered), 64)
        self.assertIn(
            {"owner_type": {"$eq": "official"}},
            where["$and"],
        )
        self.assertIn(
            {"platform": {"$eq": "Fari"}},
            where["$and"],
        )

    def test_document_metadata_gate_rejects_wrong_access_scope(self) -> None:
        item = self.official_eval_set["items"][0]
        metadata = {
            "owner_type": "official",
            "source_domain": "www.ingendynamics.com",
            "access_scope": "public",
            "confidentiality": "public",
            "is_current": True,
            "platform": item["platform"],
        }
        document = SimpleNamespace(metadata=metadata)
        self.assertTrue(
            pipeline.document_passes_metadata_gate(
                document, item, self.official_config
            )
        )
        document.metadata["access_scope"] = "internship_private"
        self.assertFalse(
            pipeline.document_passes_metadata_gate(
                document, item, self.official_config
            )
        )

    def test_blind_assets_are_frozen_and_balanced(self) -> None:
        result = pipeline.validate_assets(
            self.official_kb,
            self.blind_eval_set,
            self.blind_config,
        )
        self.assertEqual(result["evaluation_items"], 8)
        self.assertEqual(result["platform_counts"], {"Fari": 4, "Senpai": 4})
        self.assertEqual(result["expected_generation_rows"], 16)

    def test_cross_encoder_can_change_dense_ranking(self) -> None:
        config = copy.deepcopy(self.config)
        config["retrieval"]["metadata_gate"] = None
        config["retrieval"]["retriever"].update(
            {
                "fetch_k": 2,
                "top_k": 1,
                "relevance_score_threshold": 0.0,
                "auto_merge_min_children": 0,
            }
        )
        documents, _ = pipeline.retrieve_item(
            self.eval_set["items"][0],
            FakeRerankVectorStore(),
            config,
            reranker=FakeReranker(),
        )
        self.assertEqual(documents[0].metadata["chunk_id"], "B")
        self.assertEqual(documents[0].metadata["dense_relevance_score"], 0.8)
        self.assertEqual(documents[0].metadata["rerank_score"], 0.9)

    def test_candidate_messages_do_not_leak_answer_key(self) -> None:
        item = self.eval_set["items"][0]
        messages = pipeline.render_candidate_messages(item, "base", [])
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn(item["question"], rendered)
        self.assertNotIn(item["reference_answer"], rendered)
        for point in item["required_points"]:
            self.assertNotIn(point["criterion"], rendered)

    def test_public_repo_is_rejected_as_chroma_location(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the public repository"):
            pipeline.ensure_external_persist_directory(
                pipeline.REPO_ROOT / "phase_b_evaluation" / "chroma"
            )
        private_temp_parent = (
            pipeline.REPO_ROOT.parent / "private" / "phase_b_evaluation"
        )
        resolved = pipeline.ensure_external_persist_directory(private_temp_parent)
        self.assertTrue(resolved.exists())

    def test_paired_run_inputs_are_blind_and_well_formed(self) -> None:
        rows = pipeline.build_run_inputs(
            self.kb,
            self.eval_set,
            FakeVectorStore(),
            self.config,
        )
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertNotIn("reference_answer", row)
            self.assertNotIn("required_points", row)
            if row["condition"] == "base":
                self.assertEqual(row["retrieved_contexts"], [])
            else:
                self.assertTrue(row["retrieved_contexts"])
        validated = generation.validate_run_inputs(rows, self.config)
        self.assertEqual(validated["paired_eval_ids"], 6)
        self.assertTrue(validated["revisions_frozen"])

    def test_review_join_adds_hidden_rubric_after_generation(self) -> None:
        inputs = pipeline.build_run_inputs(
            self.kb,
            self.eval_set,
            FakeVectorStore(),
            self.config,
        )
        completed = []
        for row in inputs:
            output = dict(row)
            output.update(
                {
                    "status": "completed",
                    "candidate_output": "Synthetic test answer.",
                    "runtime": {"model_revision": "a" * 40},
                }
            )
            completed.append(output)
        validation = evaluation.validate_generations(
            completed,
            self.eval_set,
            self.config,
        )
        self.assertEqual(validation["rows"], 12)
        review = evaluation.prepare_review_records(completed, self.eval_set)
        self.assertEqual(len(review), 12)
        self.assertTrue(review[0]["required_points"])
        self.assertEqual(
            review[0]["point_score_template"]["status"],
            "pending_independent_judge_and_human_review",
        )

    def test_generation_accepts_legacy_tensor_chat_template_output(self) -> None:
        legacy_tensor = SimpleNamespace(shape=(1, 3))
        self.assertIs(generation.extract_input_ids(legacy_tensor), legacy_tensor)

    def test_generation_accepts_transformers_5_batch_encoding(self) -> None:
        input_ids = SimpleNamespace(shape=(1, 3))
        batch_encoding = SimpleNamespace(input_ids=input_ids)
        self.assertIs(
            generation.extract_input_ids(batch_encoding),
            input_ids,
        )


if __name__ == "__main__":
    unittest.main()
