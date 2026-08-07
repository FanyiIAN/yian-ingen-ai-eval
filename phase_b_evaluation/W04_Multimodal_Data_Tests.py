import collections
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from W04_Multimodal_Data import (
    apply_condition,
    deterministic_condition_seed,
    load_yaml,
    pixel_sha256,
    standardize_image,
    validate_spec,
)
from W04_Text_Robustness_Runner import read_jsonl


HERE = Path(__file__).resolve().parent
SPEC = HERE / "W04_Multimodal_Scenarios_v0.1.0.yaml"
INPUTS = HERE / "W04_Multimodal_Inputs_v0.1.0.jsonl"


class MultimodalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_yaml(SPEC)

    def test_scenario_spec_has_twenty_unique_balanced_images(self):
        validate_spec(self.spec)
        scenarios = self.spec["scenarios"]
        self.assertEqual(len({row["image_id"] for row in scenarios}), 20)
        self.assertEqual(
            collections.Counter(row["platform"] for row in scenarios),
            {"Aido_Rover": 10, "Sentinel_Prime_AI": 10},
        )

    def test_standardization_preserves_aspect_and_resolution(self):
        source = Image.new("RGB", (400, 200), (10, 20, 30))
        output = standardize_image(
            source,
            width=768,
            height=768,
            padding_rgb=(127, 127, 127),
        )
        self.assertEqual(output.size, (768, 768))
        self.assertEqual(output.getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(output.getpixel((384, 384)), (10, 20, 30))

    def test_noise_is_deterministic_and_brightness_is_one_factor(self):
        array = np.full((32, 32, 3), 180, dtype=np.uint8)
        clean = Image.fromarray(array, mode="RGB")
        noise_condition = self.spec["conditions"][1]
        seed = deterministic_condition_seed(42, "scenario", "noise")
        first = apply_condition(clean, noise_condition, seed=seed)
        second = apply_condition(clean, noise_condition, seed=seed)
        self.assertEqual(pixel_sha256(first), pixel_sha256(second))
        self.assertNotEqual(pixel_sha256(first), pixel_sha256(clean))
        dark = apply_condition(clean, self.spec["conditions"][2], seed=seed)
        self.assertLess(np.asarray(dark).mean(), np.asarray(clean).mean())

    def test_frozen_inputs_if_present(self):
        if not INPUTS.exists():
            self.skipTest("generated input bank not present")
        rows = read_jsonl(INPUTS)
        self.assertEqual(len(rows), 60)
        self.assertEqual(
            collections.Counter(row["condition_id"] for row in rows),
            {"clean": 20, "gaussian_noise_std_0.08": 20, "brightness_0.60": 20},
        )
        self.assertEqual(len({row["request_base_id"] for row in rows}), 60)


if __name__ == "__main__":
    unittest.main()

