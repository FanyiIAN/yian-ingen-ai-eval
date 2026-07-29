from __future__ import annotations

import unittest

from W03_RAG_Multi_Model_Generation import adapt_messages


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


if __name__ == "__main__":
    unittest.main()
