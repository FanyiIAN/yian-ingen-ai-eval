import copy
import json
import unittest
from collections import UserDict
from pathlib import Path

from W04_Multimodal_Runner import (
    json_safe_metadata,
    load_and_perturb_image,
    resolve_image_path,
    runner_architecture,
    validate_inputs,
)
from W04_Text_Robustness_Runner import load_yaml, read_jsonl


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "W04_Multimodal_Inputs_v0.1.0.jsonl"
CONFIG = HERE / "W04_Multimodal_Run_Config_v0.1.0.yaml"


class MultimodalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(INPUTS)
        cls.config = load_yaml(CONFIG)

    def test_frozen_bank_validates(self):
        result = validate_inputs(self.rows, self.config, INPUTS)
        self.assertEqual(result["row_count"], 60)
        self.assertEqual(result["scenario_count"], 20)

    def test_every_processed_pixel_hash_reproduces(self):
        for row in self.rows:
            image = load_and_perturb_image(row, INPUTS)
            self.assertEqual(image.size, (768, 768))

    def test_path_escape_is_rejected(self):
        row = copy.deepcopy(self.rows[0])
        row["image_path"] = "../../outside.jpg"
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolve_image_path(row, INPUTS)

    def test_input_hash_mismatch_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["benchmark"]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash differs"):
            validate_inputs(self.rows, config, INPUTS)

    def test_legacy_idefics_config_resolves_without_mutation(self):
        self.assertEqual(runner_architecture(self.config), "idefics2")

    def test_llava_adapter_is_explicit(self):
        config = copy.deepcopy(self.config)
        config["candidate_model"]["model_key"] = "llava_1_5_7b_hf"
        config["candidate_model"]["runner_architecture"] = "llava"
        self.assertEqual(runner_architecture(config), "llava")

    def test_library_metadata_mapping_is_json_safe(self):
        normalized = json_safe_metadata(
            UserDict({"height": 336, "width": 336, "longest_edge": None})
        )
        self.assertEqual(
            normalized,
            {"height": 336, "width": 336, "longest_edge": None},
        )
        self.assertEqual(json.loads(json.dumps(normalized)), normalized)

    def test_unknown_adapter_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["candidate_model"]["runner_architecture"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            runner_architecture(config)


if __name__ == "__main__":
    unittest.main()
