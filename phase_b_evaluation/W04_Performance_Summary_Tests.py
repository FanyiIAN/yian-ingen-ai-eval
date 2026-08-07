import unittest

from W04_Performance_Summary import (
    flatten_for_csv,
    static_runtime_row,
    summarize_events,
)


def event(run_id, condition="clean", rag=False):
    return {
        "run_item_id": run_id,
        "candidate_model_key": "model",
        "evaluation_family": "multimodal_robustness",
        "condition_id": condition,
        "prompt_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "retrieval": {
            "enabled": rag,
            "rag_total_ms": 5.0 if rag else None,
            "reason": None if rag else "not used",
        },
        "request_profile": {
            "error": None,
            "timings": {
                "input_load_ms": 2.0,
                "image_perturb_ms": 3.0,
                "prompt_build_ms": 1.0,
                "preprocess_ms": 4.0,
                "ttft_ms": 20.0,
                "generation_ms": 100.0,
                "decode_ms": 2.0,
                "question_to_response_ms": 112.5,
            },
            "resources": {
                "availability": {},
                "process_rss_mib": {"peak": 100.0},
                "system_memory_used_mib": {"peak": 200.0},
                "gpu_device_memory_used_mib": {"peak": 300.0},
                "torch_cuda": {
                    "allocated_peak_mib": 250.0,
                    "reserved_peak_mib": 280.0,
                },
                "gpu_utilization_pct": {"mean": 50.0, "peak": 90.0},
                "gpu_power_w": {"mean": 100.0, "peak": 120.0},
            },
        },
    }


class PerformanceSummaryTests(unittest.TestCase):
    def test_aggregates_latency_resources_and_throughput(self):
        summary = summarize_events([event("one"), event("two")])
        group = summary["groups"][0]
        self.assertEqual(group["row_count"], 2)
        self.assertEqual(group["latency_ms"]["question_to_response_ms"]["p95"], 112.5)
        self.assertEqual(group["resources"]["gpu_device_memory_used_peak_mib"]["max"], 300.0)
        self.assertEqual(group["tokens"]["prompt_tokens"]["p50"], 100.0)
        self.assertEqual(group["tokens"]["output_tokens"]["p50"], 20.0)
        self.assertEqual(group["tokens"]["total_tokens"]["p50"], 120.0)
        self.assertEqual(group["output_tokens_per_generation_second"]["mean"], 200.0)
        self.assertEqual(group["rag_not_applicable_row_count"], 2)

    def test_missing_rag_latency_is_not_zero(self):
        group = summarize_events([event("one")])["groups"][0]
        self.assertEqual(group["latency_ms"]["retrieval_total_ms"]["count"], 0)
        self.assertIsNone(group["latency_ms"]["retrieval_total_ms"]["mean"])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_events([event("same"), event("same")])

    def test_csv_flattening_preserves_group_identity(self):
        summary = summarize_events([event("one")])
        rows = flatten_for_csv(summary)
        self.assertTrue(rows)
        self.assertTrue(all(row["candidate_model_key"] == "model" for row in rows))
        self.assertTrue(any(row["metric_group"] == "tokens" for row in rows))

    def test_rag_runner_schema_uses_condition_fallback(self):
        row = event("legacy-rag", condition="rag", rag=False)
        row.pop("evaluation_family")
        row.pop("condition_id")
        row["condition"] = "rag"

        group = summarize_events([row])["groups"][0]

        self.assertEqual(group["evaluation_family"], "rag_performance")
        self.assertEqual(group["condition_id"], "rag")
        self.assertEqual(group["rag_enabled_row_count"], 1)

    def test_static_runtime_summary_omits_checkpoint_path_and_keeps_cost(self):
        row = static_runtime_row(
            {
                "model_key": "model",
                "model_id": "org/model",
                "model_revision": "a" * 40,
                "precision": "bfloat16",
                "model_directory": "/private/checkpoint/path",
                "model_load_ms": 1234.5,
                "model_load_resources": {
                    "gpu_device_memory_used_mib": {"peak": 456.0},
                    "process_rss_mib": {"peak": 789.0},
                },
            },
            {
                "checkpoint_path": "/private/checkpoint/path",
                "checkpoint_bytes": 1024,
                "host_ram_total_mib": 2048.0,
                "gpu": {"name": "GPU", "memory_total_mib": 4096.0},
            },
            source_file="session.jsonl",
        )

        self.assertNotIn("model_directory", row)
        self.assertNotIn("checkpoint_path", row)
        self.assertEqual(row["checkpoint_bytes"], 1024)
        self.assertEqual(row["model_load_gpu_memory_peak_mib"], 456.0)
        self.assertIsNone(row["driver_version"])
        self.assertTrue(row["driver_version_unavailability_reason"])


if __name__ == "__main__":
    unittest.main()
