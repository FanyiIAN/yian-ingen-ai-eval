import json
import time
import unittest

from W04_Resource_Monitor import (
    RequestProfiler,
    environment_manifest,
    percentile,
    summarize_numeric,
    validate_jsonable,
)


class ResourceMonitorTests(unittest.TestCase):
    def test_percentile_and_summary(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        summary = summarize_numeric([1, 2, 3, 4, None])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["p50"], 2.5)
        self.assertEqual(summary["max"], 4.0)

    def test_empty_summary_uses_null_not_zero(self):
        summary = summarize_numeric([None])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["p95"])

    def test_request_profile_is_json_serializable(self):
        with RequestProfiler(
            "unit-test-request",
            sample_interval_s=0.01,
            nvidia_smi_interval_s=0.01,
            nvidia_smi_executable="definitely-not-a-real-nvidia-smi",
            enable_torch_probe=False,
        ) as profiler:
            with profiler.stage("preprocess_ms"):
                time.sleep(0.012)
            profiler.mark_first_token()
            with profiler.stage("generation_ms"):
                time.sleep(0.012)

        result = profiler.result()
        validate_jsonable(result)
        self.assertTrue(result["measurement_availability"]["ttft_ms"]["available"])
        self.assertIsNone(result["measurement_availability"]["ttft_ms"]["reason"])
        json.dumps(result, allow_nan=False)
        self.assertGreater(result["timings"]["preprocess_ms"], 0)
        self.assertGreater(result["timings"]["generation_ms"], 0)
        self.assertGreaterEqual(
            result["timings"]["question_to_response_ms"],
            result["timings"]["generation_ms"],
        )
        self.assertIsNotNone(result["timings"]["ttft_ms"])
        self.assertGreaterEqual(result["resources"]["host_sample_count"], 2)
        self.assertFalse(
            result["resources"]["availability"]["nvidia_smi"]["available"]
        )
        self.assertIsNotNone(
            result["resources"]["availability"]["nvidia_smi"]["reason"]
        )

    def test_repeated_stage_names_are_summed_and_traced(self):
        with RequestProfiler(
            "repeat-stage",
            sample_interval_s=0.01,
            nvidia_smi_interval_s=0.02,
            nvidia_smi_executable="missing-nvidia-smi",
            enable_torch_probe=False,
        ) as profiler:
            with profiler.stage("input_load_ms"):
                time.sleep(0.004)
            with profiler.stage("input_load_ms"):
                time.sleep(0.004)
        result = profiler.result()
        self.assertFalse(result["measurement_availability"]["ttft_ms"]["available"])
        self.assertIsNotNone(result["measurement_availability"]["ttft_ms"]["reason"])
        records = result["timings"]["stage_records"]
        self.assertEqual(len(records), 2)
        self.assertGreaterEqual(
            result["timings"]["input_load_ms"],
            sum(record["duration_ms"] for record in records) - 0.001,
        )

    def test_stage_name_contract(self):
        with RequestProfiler(
            "bad-stage",
            sample_interval_s=0.01,
            nvidia_smi_interval_s=0.02,
            nvidia_smi_executable="missing-nvidia-smi",
            enable_torch_probe=False,
        ) as profiler:
            with self.assertRaises(ValueError):
                with profiler.stage("preprocess"):
                    pass

    def test_environment_manifest_preserves_unavailable_gpu(self):
        manifest = environment_manifest(import_heavy_modules=False)
        validate_jsonable(manifest)
        self.assertIn("python", manifest)
        self.assertIn("nvidia_smi", manifest["availability"])


if __name__ == "__main__":
    unittest.main()
