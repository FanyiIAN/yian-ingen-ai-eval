# Week 7 AI Model Evaluation Dashboard

Stakeholder-facing Streamlit dashboard for the frozen Weeks 2–6 public/synthetic evaluation evidence. It provides three required views—Model Scorecard, RAG Performance, and Robustness Snapshot—and audience-specific information for an AI evaluation engineer, product manager, and executive.

The audience modes are intentionally different:

- **Executive:** exactly three decision indicators and one recommended action; no technical tabs.
- **Product manager:** two focused tabs, `Platform Risk` and `RAG Readiness`.
- **AI evaluation engineer:** the complete scorecard, RAG frontier, robustness, source manifest, and reproduction commands.

## One-command launch

The dashboard supports Python 3.9 or newer. The first launch requires network access to install the three pinned dependency families into an ignored local `.venv`.

From a fresh clone on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File phase_d_capstone/W07_Dashboard/run_dashboard.ps1
```

On Linux or macOS:

```bash
bash phase_d_capstone/W07_Dashboard/run_dashboard.sh
```

The launch script creates an ignored local `.venv`, installs the three pinned dependency families, and starts Streamlit. It does not download model weights or run evaluation.

## Data contract

The app reads only the eleven committed CSV files under `data/`. It never reads row-level experiment artifacts and performs no aggregation, model inference, retrieval, scoring, or LLM Judge call at launch. Interactive filtering and chart presentation are the only runtime operations.

To intentionally rebuild the CSV presentation layer after reviewing a source change:

```bash
python phase_d_capstone/W07_Dashboard/build_dashboard_data.py
python -m unittest phase_d_capstone.W07_Dashboard.test_dashboard_contract -v
```

`data_manifest.csv` records the path, byte count, and SHA-256 of every registered Week 2–6 input.

`dashboard_metadata.csv` freezes cross-view result constants such as the Judge calibration gate, seed, RAG cell count, and VLM sample counts. This prevents the Streamlit presentation layer from embedding result numbers in application code.

The launch scripts change into the dashboard directory so `.streamlit/config.toml` is always applied. That file explicitly selects a light theme; application CSS additionally fixes metric labels, tabs, captions, and widget labels to accessible dark text on the light content background.

## Interpretation boundary

The dashboard's readiness values are **diagnostic proxies**, calculated as the mean of severity-weighted Task Accuracy and Contextual Grounding divided by the five-point scale. Coverage is reported separately. The Week 2 Judge missed its preregistered human-label calibration gate (`α=0.7551 < 0.80`), so the dashboard does not provide a validated model ranking, product certification, or deployed PIC readiness measurement.

RAG, robustness, and VLM quality metrics are also AI-assisted diagnostics without independent human calibration. All views retain model revision, evaluation-set version, seed, evidence status, and scope information in the precomputed data layer.

## Troubleshooting

- Confirm Python 3.9 or newer is available as `python` (Windows) or `python3` (Linux/macOS).
- If port 8501 is occupied, append Streamlit arguments after the script command or launch manually with `python -m streamlit run phase_d_capstone/W07_Dashboard/app.py --server.port 8502`.
- If source hashes change unexpectedly, do not overwrite the committed CSVs until the upstream result version and evidence status have been reviewed.

## Verification

Static data and source contracts:

```bash
python -m unittest phase_d_capstone.W07_Dashboard.test_dashboard_contract -v
```

Runtime rendering for all three personas, after the launcher has created `.venv`:

```bash
phase_d_capstone/W07_Dashboard/.venv/Scripts/python.exe -m unittest phase_d_capstone.W07_Dashboard.test_dashboard_app -v
```

On Linux/macOS use `.venv/bin/python` instead. The earlier clean-copy dependency-install verification is retained in `fresh_copy_verification_v1.1.0.json` as a superseded UI record. The current `fresh_copy_verification_v1.2.0.json` binds the 18/18 contract result, three refreshed screenshots, screenshot hashes, and a headless browser check with zero white-on-white text candidates. The record explicitly does not claim an independent human usability study.
