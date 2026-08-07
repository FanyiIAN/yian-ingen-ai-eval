import json
import unittest
from pathlib import Path

from W04_Text_Robustness_Runner import (
    FirstTokenProfilerStreamer,
    candidate_model,
    event_paths,
    load_yaml,
    materialize_views,
    read_jsonl,
    render_candidate_prompt,
    validate_input_bank,
)


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "W04_Robustness_Inputs_v0.1.0.jsonl"
CONFIG = HERE / "W04_Robustness_Run_Config_v0.1.0.yaml"
PROMPT = HERE.parent / "phase_a_design" / "W02_Prompt_Spec_v0.4.0.yaml"


class FakeProfiler:
    def __init__(self):
        self.calls = 0

    def mark_first_token(self):
        self.calls += 1


class TextRobustnessRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(INPUTS)
        cls.config = load_yaml(CONFIG)
        cls.prompt = load_yaml(PROMPT)

    def test_frozen_input_validates(self):
        result = validate_input_bank(self.rows, self.config, INPUTS)
        self.assertEqual(result["row_count"], 182)

    def test_all_three_models_have_frozen_revisions(self):
        for key in (
            "flan_t5_base",
            "mistral_7b_instruct_v0_2",
            "llama31_8b_instruct",
        ):
            self.assertEqual(len(candidate_model(self.config, key)["revision"]), 40)

    def test_render_uses_variant_not_source_scenario(self):
        row = next(
            value
            for value in self.rows
            if value.get("variant_type") == "tone_shift"
        )
        rendered = render_candidate_prompt(row, self.prompt)
        self.assertIn(row["input_stimulus"], rendered)
        self.assertIn("SYSTEM POLICY", rendered)

    def test_streamer_marks_only_first_generated_put(self):
        profiler = FakeProfiler()
        streamer = FirstTokenProfilerStreamer(profiler)
        streamer.put([1, 2])  # initial prompt/decoder IDs
        self.assertEqual(profiler.calls, 0)
        streamer.put([3])
        streamer.put([4])
        self.assertEqual(profiler.calls, 1)
        self.assertEqual(streamer.generated_put_count, 2)

    def test_materialized_views_preserve_join_key(self):
        event = {
            "run_item_id": "model::request",
            "request_base_id": "request",
            "candidate_model_key": "model",
            "candidate_model_id": "example/model",
            "candidate_model_revision": "a" * 40,
            "evaluation_family": "semantic_robustness",
            "scenario_id": "FARI-001",
            "variant_type": "original",
            "mask_ratio": None,
            "cold_or_warm": "warm_steady_state",
            "request_profile": {"timings": {}, "resources": {}},
            "prompt_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "candidate_output": "Stop.",
        }
        paths = event_paths(HERE, "_w04_test_materialized_view")
        created = [paths["candidates"], paths["traces"]]
        try:
            materialize_views([event], paths)
            candidate = read_jsonl(paths["candidates"])[0]
            trace = read_jsonl(paths["traces"])[0]
        finally:
            for path in created:
                path.unlink(missing_ok=True)
        self.assertEqual(candidate["run_item_id"], trace["run_item_id"])
        self.assertNotIn("request_profile", candidate)
        self.assertIn("request_profile", trace)


if __name__ == "__main__":
    unittest.main()
