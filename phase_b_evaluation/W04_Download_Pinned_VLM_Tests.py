import json
import shutil
import unittest
from pathlib import Path

from W04_Download_Pinned_VLM import load_model_contract, validate_snapshot


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "W04_Multimodal_Run_Config_LLaVA_v0.2.0.yaml"


class DownloadPinnedVlmTests(unittest.TestCase):
    def test_llava_contract_is_fully_pinned(self):
        contract = load_model_contract(CONFIG)
        self.assertEqual(contract["model_id"], "llava-hf/llava-1.5-7b-hf")
        self.assertEqual(len(contract["revision"]), 40)
        self.assertEqual(contract["runner_architecture"], "llava")

    def test_snapshot_validation_checks_all_shards(self):
        # tempfile uses a restrictive Windows ACL that conflicts with some
        # sandboxed CI identities, so use an inherited-ACL repository folder.
        root = HERE / ".w04_download_test_snapshot"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing VLM weight shards"):
                validate_snapshot(root)
            (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            result = validate_snapshot(root)
            self.assertEqual(result["weight_shard_count"], 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
