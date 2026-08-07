from __future__ import annotations

import unittest

from W03_RAG_Multi_Model_Generation import adapt_messages, host_memory_snapshot


MESSAGES = [
    {"role": "system", "content": "Use only evidence."},
    {"role": "user", "content": "CONTEXT\nA\n\nQUESTION\nQ?"},
]


class MultiModelGenerationTests(unittest.TestCase):
    def test_native_chat_preserves_messages(self) -> None:
        self.assertEqual(
            MESSAGES,
            adapt_messages(MESSAGES, "native_chat"),
        )

    def test_mistral_adapter_folds_system_into_user(self) -> None:
        adapted = adapt_messages(MESSAGES, "fold_system_into_user")
        self.assertEqual(1, len(adapted))
        self.assertEqual("user", adapted[0]["role"])
        self.assertIn("Use only evidence.", adapted[0]["content"])
        self.assertIn("QUESTION", adapted[0]["content"])

    def test_seq2seq_adapter_is_explicit_text_to_text(self) -> None:
        adapted = adapt_messages(MESSAGES, "seq2seq_text")
        self.assertIsInstance(adapted, str)
        self.assertTrue(adapted.startswith("TASK\nAnswer the question"))
        self.assertIn("SYSTEM INSTRUCTIONS", adapted)
        self.assertIn("Use only evidence.", adapted)
        self.assertIn("QUESTION", adapted)
        self.assertTrue(adapted.endswith("FINAL ANSWER\n"))

    def test_unknown_adapter_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported runtime adapter"):
            adapt_messages(MESSAGES, "unknown")

    def test_host_memory_snapshot_has_json_safe_nonnegative_counters(self) -> None:
        snapshot = host_memory_snapshot()
        self.assertEqual(
            {
                "process_rss_bytes",
                "process_peak_rss_bytes",
                "system_used_bytes",
            },
            set(snapshot),
        )
        for value in snapshot.values():
            self.assertTrue(value is None or value >= 0)


if __name__ == "__main__":
    unittest.main()
