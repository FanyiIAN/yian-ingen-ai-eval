"""Build the public-safe Week 8 evidence registry from frozen repository files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase_d_capstone" / "W08_Evidence_Registry_v1.0.0.json"

SOURCES = [
    ("phase_a_design/W02_Scenarios.yaml", "35-scenario benchmark"),
    ("phase_a_design/W02_Baseline_Eval_Results.csv", "70-row baseline evidence"),
    ("phase_a_design/W02_Baseline_Agreement.json", "Judge prompt agreement"),
    ("phase_b_evaluation/W03_Three_Model_Diagnostic_Summary.json", "105-row three-model summary"),
    ("phase_b_evaluation/W03_RAG_Long_Source_Summary_v1.0.0.json", "corrected long-source RAG summary"),
    ("phase_b_evaluation/W04_Robustness_Summary_v0.1.0.json", "semantic and masked-input robustness"),
    ("phase_b_evaluation/W04_Multimodal_Architecture_Comparison_v0.2.0.json", "two-VLM comparison"),
    ("phase_b_evaluation/W04_System_Performance_Summary_v0.2.0.json", "latency and resource evidence"),
    ("phase_c_synthesis/W05_RAG_Long_Source_Optimisation_Summary_v1.1.0.json", "18-cell RAG factorial"),
    ("phase_c_synthesis/W05_PIC20_Model_Analysis.md", "six-class PIC analysis"),
    ("phase_c_synthesis/W06_Claim_Evidence_Matrix_v1.0.0.csv", "Week 6 claim controls"),
    ("phase_c_synthesis/W06_Evidence_Registry_v1.0.0.json", "Week 6 frozen source registry"),
    ("phase_d_capstone/W07_Dashboard/data/data_manifest.csv", "dashboard source hashes"),
    ("phase_d_capstone/W07_Dashboard/data/dashboard_metadata.csv", "dashboard version and gates"),
    ("phase_d_capstone/W08_Claim_Evidence_Matrix_v1.0.0.csv", "capstone claim matrix"),
    ("phase_d_capstone/W08_Capstone_Report_Source.md", "capstone narrative source"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for relative_path, role in SOURCES:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        rows.append(
            {
                "path": relative_path,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    registry = {
        "registry_version": "1.0.0",
        "generated_date": "2026-08-19",
        "source_count": len(rows),
        "evidence_boundary": (
            "Public/synthetic diagnostic evidence only; no deployed InGen product, "
            "proprietary PIC runtime, customer data, or confidential source material."
        ),
        "latest_only_controls": {
            "rag_knowledge_base": "W03 long-source v1.0.0",
            "rag_optimisation": "W05 long-source v1.1.0",
            "dashboard": "v1.2.0",
            "superseded_atomic_section_results_used_for_current_claims": False,
        },
        "sources": rows,
    }
    OUT.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(rows)} verified sources")


if __name__ == "__main__":
    main()
