# Week 8 Capstone Submission Index

## Reference-required deliverables

| Artifact | Purpose | Status |
|---|---|---|
| `W08_Capstone_Report.docx` | Nineteen-page English capstone report with ten required sections, a target-versus-achieved table and recommendation in every section, figures, traceability, and limitations | Complete; rendered page-by-page |
| `W08_Capstone_Deck.pptx` | Twelve-slide, finding-led English final readout with a number in every headline and repository-relative evidence notes | Complete; 12/12 slides rendered and overflow-tested |
| `W08_Capstone_Deck_Speaker_Script.md` | Slide-matched 30-minute speaking plan and source list for the final readout | Complete |
| `W08_Retrospective.md` | One-page reflection on the most surprising finding, weakest section, and a 12-week extension | Complete |
| `W08_Final_Evaluation_Rubric.md` | Intern evidence-based self-assessment and blank supervisor scoring/signature fields | Intern portion complete; supervisor action pending |
| `../weekly/Wk-08-Final-EvalLog.md` | Final evaluated/found/mechanism/next-action record | Complete |

## Supplemental final project review

| Artifact | Purpose | Status |
|---|---|---|
| `W08_Final_15min_Week7_8_Project_Review_Deck_EN.pptx` | Fifteen-slide English review that devotes Slides 8-15 to the new Week 7 dashboard and Week 8 capstone, while Slides 1-7 synthesise Weeks 1-6 | Complete; 15/15 slides rendered and overflow-tested |
| `W08_Final_15min_Week7_8_Project_Review_Script_EN.md` | Slide-matched English script with an exact 15-minute timing plan and repository-relative sources | Complete; 15 sections and 900 seconds |

This shorter package is a stakeholder-specific adaptation. It supplements rather
than replaces the reference-required twelve-slide, thirty-minute final readout.

## Evidence and defence controls

| Artifact | Purpose | Status |
|---|---|---|
| `W08_Capstone_Report_Source.md` | Auditable English source for the generated DOCX | Complete |
| `W08_Claim_Evidence_Matrix_v1.0.0.csv` | Twelve principal claims mapped to exact revision, evaluation set, seed, scope, and source | Complete |
| `W08_Evidence_Registry_v1.0.0.json` | SHA-256 registry for the current evidence sources | Complete; 16 verified sources |
| `W08_Final_Readout_QA.md` | Defence notes for the 35 scenarios, RAGAS-style metrics, masked input, severity weights, and validity boundaries | Complete |
| `W08_Capstone_Contract_Tests.py` | Deliverable, section, slide-count, traceability, retrospective, and privacy contracts | Complete |
| `W08_Finalise_Evidence.py` | Deterministic evidence-registry generator | Complete |

## Verification commands

From the repository root:

```powershell
python phase_d_capstone/W08_Finalise_Evidence.py
python -m unittest phase_d_capstone.W08_Capstone_Contract_Tests -v
python -m unittest phase_d_capstone.W07_Dashboard.test_dashboard_contract -v
phase_d_capstone/W07_Dashboard/.venv/Scripts/python.exe -m unittest phase_d_capstone.W07_Dashboard.test_dashboard_app -v
python phase_c_synthesis/W06_Evidence_Synthesis.py
python -m unittest phase_c_synthesis.W06_Evidence_Synthesis_Tests -v
```

The final DOCX was exported through Microsoft Word and inspected across all 19 pages. The reference-required final PPTX was rendered from the saved file, inspected across all 12 slides, and passed the presentation overflow test. The supplemental 15-minute PPTX was likewise inspected across all 15 slides, passed the overflow test, and contains a repository-relative source block in every speaker-notes page. Those visual-QA images are private working evidence rather than repository deliverables.

## Interpretation boundary

The capstone provides reproducible public/synthetic diagnostic evidence and a sequence of validation gates. It does not certify a deployed InGen product, validate proprietary PIC runtime readiness, estimate field failure rates, or replace independent domain review.

## External completion items

- Deliver the planned 30-minute readout and 15-minute Q&A.
- Obtain supervisor scores, written feedback, and both signatures in `W08_Final_Evaluation_Rubric.md`.
- Push the reviewed release/tag only after repository-owner approval.
