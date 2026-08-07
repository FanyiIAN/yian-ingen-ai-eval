from __future__ import annotations

import unittest
from pathlib import Path

from W03_RAG_Expanded_MultiModel_Run import (
    artifact_paths,
)


class ExpandedMultiModelRunTests(unittest.TestCase):
    def test_artifacts_stay_below_immutable_run_directory(self) -> None:
        root = Path("/workspace/w03-run")
        paths = artifact_paths(root)
        self.assertTrue(all(root in path.parents for path in paths.values()))

if __name__ == "__main__":
    unittest.main()
