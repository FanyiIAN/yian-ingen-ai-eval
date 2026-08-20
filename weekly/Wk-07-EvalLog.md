# Week 7 Evaluation Log

- **Phase:** D — evaluation dashboard and stakeholder communication
- **Dashboard version:** `1.2.0` persona and contrast correction
- **Evidence:** frozen public/synthetic Weeks 2–6 outputs
**Seed:** `42` for every cited model run

## Work completed

- Audited the Week 7 reference objective, required views, three personas, deliverables, tools, and self-checks.
- Registered eight Week 2–6 source artifacts and produced eleven dashboard-specific CSVs with source paths, byte counts, and SHA-256 hashes.
- Implemented distinct audience architectures: Executive has three indicators and no technical tabs; Product Manager has `Platform Risk` and `RAG Readiness`; Engineer has the complete scorecard, RAG, robustness, and data-source audit views.
- Added a Windows and POSIX one-command launcher, isolated `.venv` workflow, requirements file, and launch README.
- Added twelve automated source/data contract tests covering CSV-only launch, five-platform/three-model coverage, proxy formula, top-three concerns, 18-cell/three-Pareto logic, robustness/VLM row contracts, exact executive indicators, metadata binding, personas, views, deterministic rebuilding, launch error propagation, and launch assets.
- Added six Streamlit application tests covering persona-specific structures, all interactive choices, seven chart payloads, audit wording, explicit light-theme registration, and masked-curve color separation.
- User browser review exposed white-on-light labels and overly similar persona pages in v1.1. Version 1.2 fixes both; all three screenshots were refreshed from the current build and hash-bound in `fresh_copy_verification_v1.2.0.json`.
- Re-ran the Windows launcher from a clean copied package with no pre-existing `.venv`; dependency installation completed and the independent Streamlit health endpoint returned `ok`.

## Evaluation design decisions

1. **Precompute every metric.** The app performs no raw aggregation, inference, retrieval, or scoring. Presentation-time filtering is the only data operation.
2. **Keep readiness diagnostic.** The reference requires readiness scores, but the Week 2 Judge calibration failed. Every score therefore says `diagnostic`, shows coverage, and sits behind a visible interpretation banner.
3. **Separate consistency from correctness.** Robustness charts show stable-failure counts with consistency so a model cannot receive a favorable visual for failing consistently.
4. **Show the full Pareto set.** The RAG view retains all 18 configurations and highlights three non-dominated points instead of hiding trade-offs behind one recommendation.
5. **Make executive action evidence-aware.** The executive view recommends independent calibration and simulator validation rather than immediate model selection.

## Problems and resolutions

| Problem | Resolution | Remaining boundary |
|---|---|---|
| The word `readiness` could imply deployment approval | Renamed every numeric result as a diagnostic proxy and added an above-the-fold claim-boundary banner | Stakeholders must still be briefed on failed calibration |
| The default Streamlit browser theme overrode Plotly chart styling | Added explicit paper, plot, grid, axis, and font colors to every figure | Only one desktop viewport was visually checked |
| Base RAG answers have no faithfulness value | Kept Base faithfulness structurally empty and avoided comparing unequal-component proxy scores | RAG quality remains uncalibrated |
| Failure summaries contain `unresolved`, which is evaluator uncertainty | Retained it in the heat map but excluded it from behavioral top-three concern cards | A severity-by-failure joint table would support stronger risk ranking |
| Initial dependency installation exceeded the command time window | Verified the isolated environment after completion; all three packages were installed and the app launched | Fresh-clone installation still requires network access |
| Cross-view result numbers such as Judge α and sample counts were embedded in presentation code | Added `dashboard_metadata.csv`, generated from the frozen Week 5–6 inputs, and bound the app to it | Narrative wording remains in code, but evaluation results do not |
| Windows launcher did not forward Streamlit arguments or reliably surface pip failure | Forwarded all extra arguments and explicitly propagated dependency-install and server exit codes | First installation still depends on package-index availability |
| Native Streamlit labels inherited dark-theme white text over a custom light background | Added a project `base="light"` theme, changed the launcher working directory so the theme is loaded, added label/tab/caption/metric contrast overrides, and refreshed all three current-browser screenshots | One desktop viewport was tested; broader accessibility review remains external |
| Persona modes shared almost the same technical tabs | Replaced the shared layout with three different information architectures matched to the reference personas | Underlying findings intentionally remain identical across personas |
| Evidence tab said ten CSVs and did not explain its purpose | Renamed it `Data Sources & Reproduction`, restricted it to Engineer mode, explained SHA-256/source lineage, and corrected the count to eleven presentation CSVs and eight source artifacts | This audit view is not intended for PM or Executive use |
| Masked-input lines rendered as nearly identical black colors | Registered explicit blue, cyan, and amber model colors and added a chart-spec regression test | Color-blind accessibility has not been independently assessed |

## Verification result

- Dashboard CSVs: `11/11` present and non-empty
- Source/data contract tests: `12/12` passed
- Streamlit application tests: `6/6` passed across all persona structures and interactive choices
- Text scorecard: five platforms × three models
- RAG factorial: 18 cells, three Pareto-optimal cells, one balanced choice
- Robustness: three models and 12 masked-curve points
- VLM: two models × three conditions
- Visual status: current v1.2 screenshots cover all three personas; a headless browser traversed the personas and four Engineer tabs and found zero white-on-white text candidates
- Clean-copy launch: new `.venv`, Streamlit `1.50.0`, Plotly `6.9.0`, pandas `2.3.3`, health `ok`
- Model/Judge calls at launch: zero

## Week 7 interpretation

The dashboard is useful because it makes uncertainty visible at the same level as performance. Its main result is not the `85.8/100` diagnostic proxy; it is that the interface prevents that number from appearing without the failed-calibration warning, score coverage, observed unsafe outputs, and required validation action.

## Next step

Carry the frozen v1.2 dashboard into the Week 8 capstone. If independent product-manager access becomes available, add a human comprehension/usability result without replacing the existing automated interface evidence.
