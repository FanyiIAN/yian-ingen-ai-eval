from __future__ import annotations

import csv
import json
import re
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "phase_d_capstone"


class Week8CapstoneContractTests(unittest.TestCase):
    def test_required_deliverables_exist(self) -> None:
        required = [
            PHASE / "W08_Capstone_Report.docx",
            PHASE / "W08_Capstone_Deck.pptx",
            PHASE / "W08_Retrospective.md",
            PHASE / "W08_Final_Evaluation_Rubric.md",
            PHASE / "W08_Claim_Evidence_Matrix_v1.0.0.csv",
            PHASE / "W08_Evidence_Registry_v1.0.0.json",
            ROOT / "weekly" / "Wk-08-Final-EvalLog.md",
        ]
        for path in required:
            self.assertTrue(path.is_file() and path.stat().st_size > 0, path)

    def test_report_has_ten_required_sections_and_section_contracts(self) -> None:
        text = (PHASE / "W08_Capstone_Report_Source.md").read_text(encoding="utf-8")
        sections = re.findall(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)
        self.assertEqual([str(i) for i in range(1, 11)], [number for number, _ in sections])
        chunks = re.split(r"^## \d+\. .+$", text, flags=re.MULTILINE)[1:]
        self.assertEqual(10, len(chunks))
        for chunk in chunks:
            self.assertIn("### Metric target versus achieved", chunk)
            self.assertIn("**Recommendation:**", chunk)

    def test_deck_has_twelve_slides_and_numbered_headlines(self) -> None:
        path = PHASE / "W08_Capstone_Deck.pptx"
        with zipfile.ZipFile(path) as archive:
            slide_names = sorted(
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            )
            self.assertEqual(12, len(slide_names))
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            for slide_name in slide_names:
                root = ET.fromstring(archive.read(slide_name))
                text = " ".join(node.text or "" for node in root.findall(".//a:t", ns))
                self.assertRegex(text, r"\d", slide_name)

    def test_claim_matrix_and_registry_are_complete(self) -> None:
        with (PHASE / "W08_Claim_Evidence_Matrix_v1.0.0.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(12, len(rows))
        self.assertTrue(all(row["model_revision"] and row["evaluation_set"] and row["seed"] for row in rows))
        registry = json.loads((PHASE / "W08_Evidence_Registry_v1.0.0.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["source_count"], len(registry["sources"]))
        self.assertGreaterEqual(registry["source_count"], 16)
        self.assertFalse(registry["latest_only_controls"]["superseded_atomic_section_results_used_for_current_claims"])

    def test_retrospective_is_one_page_length_and_complete(self) -> None:
        text = (PHASE / "W08_Retrospective.md").read_text(encoding="utf-8")
        words = re.findall(r"\b[\w.-]+\b", text)
        self.assertGreaterEqual(len(words), 300)
        self.assertLessEqual(len(words), 700)
        for heading in ("Most surprising finding", "Weakest section", "12-week version"):
            self.assertIn(heading, text)

    def test_public_capstone_files_do_not_reference_private_workspace(self) -> None:
        checked = [
            PHASE / "W08_Capstone_Report_Source.md",
            PHASE / "W08_Retrospective.md",
            PHASE / "W08_Final_Readout_QA.md",
            ROOT / "weekly" / "Wk-08-Final-EvalLog.md",
        ]
        forbidden = (
            "D:" + "\\newIntern\\" + "private",
            "private" + "/references",
            "confidential source pdf",
        )
        for path in checked:
            lower = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token.lower(), lower, f"{token} in {path.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
