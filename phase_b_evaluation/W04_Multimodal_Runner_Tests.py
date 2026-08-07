import copy
import unittest
from pathlib import Path

from W04_Multimodal_Runner import (
    load_and_perturb_image,
    resolve_image_path,
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


if __name__ == "__main__":
    unittest.main()

