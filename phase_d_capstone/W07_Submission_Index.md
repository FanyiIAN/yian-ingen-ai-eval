# Week 7 Submission Index

## Required deliverables

| Artifact | Purpose | Status |
|---|---|---|
| `W07_Dashboard/` | Streamlit app, eleven precomputed CSVs, requirements, cross-platform launch scripts, 18 static/runtime tests, refreshed v1.2 screenshots, clean-copy/visual verification, and README | Complete |
| `W07_Dashboard_Design_Doc.md` | Persona rationale, three required views, annotated screenshots, launch guide, and acceptance evidence | Complete |
| `../weekly/Wk-07-EvalLog.md` | Work log, design decisions, problems, verification, and next step | Complete |

## Primary commands

```powershell
powershell -ExecutionPolicy Bypass -File phase_d_capstone/W07_Dashboard/run_dashboard.ps1
python -m unittest phase_d_capstone.W07_Dashboard.test_dashboard_contract -v
phase_d_capstone/W07_Dashboard/.venv/Scripts/python.exe -m unittest phase_d_capstone.W07_Dashboard.test_dashboard_app -v
```

The Windows launcher was verified from a clean copied package with no pre-existing `.venv`; dependency installation completed and `/_stcore/health` returned `ok`. The current v1.2 UI separately passed 18/18 contracts and a headless three-persona browser check with zero white-on-white text candidates; see `W07_Dashboard/fresh_copy_verification_v1.2.0.json`.

## Claim boundary

The app is a stakeholder communication layer over frozen public/synthetic diagnostic evidence. It does not rerun inference, repair evaluator calibration, certify product safety, rank deployed InGen systems, or measure proprietary PIC runtime readiness.
