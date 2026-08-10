import copy
import unittest

from W04_Multimodal_Comparison import (
    build_comparison,
    percentile,
    validate_controlled_inputs,
)


CONDITIONS = ("clean", "gaussian_noise_std_0.08", "brightness_0.60")


def event(model_key, condition, index):
    request_id = f"request::{condition}"
    return {
        "run_item_id": f"{model_key}::{request_id}",
        "request_base_id": request_id,
        "scenario_id": "scenario-1",
        "platform": "Aido_Rover",
        "condition_id": condition,
        "condition_seed": index,
        "image_file_sha256": "a" * 64,
        "processed_pixel_sha256": chr(98 + index) * 64,
        "user_prompt_sha256": "f" * 64,
        "candidate_model_key": model_key,
        "candidate_output_sha256": chr(107 + index) * 64,
        "seed": 42,
        "cold_or_warm": "warm_steady_state",
        "output_tokens": 10,
        "request_profile": {
            "timings": {
                "question_to_response_ms": 1000.0 + index,
                "ttft_ms": 100.0 + index,
                "generation_ms": 500.0 + index,
            },
            "resources": {
                "gpu_device_memory_used_mib": {"peak": 10000.0 + index},
                "gpu_power_w": {"peak": 200.0 + index},
            },
        },
    }


def scored(model_key, condition, index, total):
    request_id = f"request::{condition}"
    output_hash = chr(107 + index) * 64
    return {
        "score_id": f"multimodal::{model_key}::{request_id}",
        "run_item_id": f"{model_key}::{request_id}",
        "request_base_id": request_id,
        "scenario_id": "scenario-1",
        "candidate_model_key": model_key,
        "candidate_output_sha256": output_hash,
        "condition_id": condition,
        "platform": "Aido_Rover",
        "score_status": "parsed",
        "normalized_score": {
            "scene_interpretation": 2,
            "decision_recommendation": 2 if total >= 4 else 1,
            "uncertainty_and_claim_control": total - 4 if total >= 4 else 1,
            "total_score": total,
            "decision_acceptable": total >= 4,
            "forbidden_claim_present": False,
            "triggered_forbidden_claims": [],
        },
        "judge": {"model_config_sha256": "j" * 64},
        "scorer_version": "0.1.2",
        "judge_method": "ai_assisted_single_pass_rubric",
    }


def session(model_key, architecture):
    return {
        "runtime": {
            "model_key": model_key,
            "model_id": f"example/{model_key}",
            "model_revision": "r" * 40,
            "runner_architecture": architecture,
            "precision": "float16",
            "attention_implementation": "sdpa",
            "processor_class": "ExampleProcessor",
            "torch_version": "2.8.0",
            "transformers_version": "5.14.1",
            "gpu_name": "NVIDIA A40",
            "model_load_ms": 1000.0,
            "model_load_resources": {
                "gpu_device_memory_used_mib": {"peak": 9000.0}
            },
        }
    }


class MultimodalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.model_a = "idefics2_8b_chatty"
        self.model_b = "llava_1_5_7b_hf"
        self.events = {
            model: [event(model, condition, index) for index, condition in enumerate(CONDITIONS)]
            for model in (self.model_a, self.model_b)
        }
        self.scores = {
            self.model_a: [
                scored(self.model_a, condition, index, 5 - min(index, 1))
                for index, condition in enumerate(CONDITIONS)
            ],
            self.model_b: [
                scored(self.model_b, condition, index, 4)
                for index, condition in enumerate(CONDITIONS)
            ],
        }
        self.sessions = {
            self.model_a: [session(self.model_a, "idefics2")],
            self.model_b: [session(self.model_b, "llava")],
        }

    def test_builds_two_model_comparison(self):
        summary = build_comparison(self.scores, self.events, self.sessions)
        self.assertEqual(summary["controlled_inputs"]["status"], "matched")
        self.assertEqual(len(summary["models"]), 2)
        self.assertEqual(summary["pairwise_clean_comparison"][0]["shared_scenario_count"], 1)

    def test_rejects_changed_pixels(self):
        events = copy.deepcopy(self.events)
        events[self.model_b][0]["processed_pixel_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "controlled input differs"):
            validate_controlled_inputs(events)

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)


if __name__ == "__main__":
    unittest.main()
