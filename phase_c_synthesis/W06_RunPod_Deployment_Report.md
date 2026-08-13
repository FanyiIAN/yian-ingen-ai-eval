# Week 6 RunPod Deployment and Reproducibility Report

**Date:** 2026-08-13  
**Pipeline:** `W06_Evidence_Synthesis.py` v1.0.0  
**Evidence registry:** `W06_Evidence_Registry_v1.0.0.json`  
**Temporary pod:** `llcx9pbzz9sizd` (`w06-evidence-repro`)

## Purpose and scope

This deployment tests whether the Week 6 evidence-synthesis pipeline can be moved from the Windows development workspace to a clean Linux RunPod environment and reproduce the same registered outputs. It is a portability and provenance test, not a model-inference, GPU-throughput, latency, or product-deployment benchmark.

Only the 12 frozen public-repository inputs named in the evidence registry, the Week 6 pipeline and tests, and the formal Week 6 Markdown artifacts were uploaded. Confidential reference PDFs, private working notes, credentials, model caches, and customer data were not transferred.

## Hardware decision and incident record

An on-demand NVIDIA A40 pod was initially requested. The pod was stopped briefly to correct the SSH public-key configuration; the A40 capacity was reclaimed before restart, and RunPod then reported no compatible A40 instance for migration. Because the Week 6 pipeline is dependency-free evidence synthesis and does not perform inference, the same pod was restarted in RunPod's CPU-only mode instead of waiting for an unrelated GPU resource.

This fallback is valid for the stated reproducibility question: output equality depends on registered inputs and deterministic standard-library code, not accelerator hardware. It does **not** provide new GPU or model-performance evidence.

## Remote environment and procedure

- Operating system: Linux `6.8.0-136-generic`, x86-64
- Python: CPython `3.12.3`
- Remote working directory: `/workspace/w06-evidence-repro/yian-ingen-ai-eval`
- Registered source count: 12
- Generated claim count: 7

The remote verification command was:

```bash
python3 phase_c_synthesis/W06_Evidence_Synthesis.py \
  --verify-only \
  --verification-record phase_c_synthesis/W06_RunPod_Deployment_Verification_v1.0.0.json
python3 -m unittest phase_c_synthesis.W06_Evidence_Synthesis_Tests
```

## Results

| Check | Result |
|---|---:|
| Registered input hash verification | 12/12 passed |
| Claim extraction | 7/7 produced |
| Contract and report tests | 10/10 passed |
| Local/remote summary SHA-256 | Match |
| Local/remote claim-matrix SHA-256 | Match |

Exact matched hashes:

- Evidence summary: `7754b84eb9f62c7e6d5d14854d14dff87e170115d0c8789fc46d52ad43db17b7`
- Claim-evidence matrix: `69981e57473363212aca63d4063616a4384a8705541f32e39cbd89159b118584`

The machine-generated environment record is retained in `W06_RunPod_Deployment_Verification_v1.0.0.json`. The matching hashes demonstrate deterministic reproduction for the registered Week 6 synthesis outputs across the tested Windows and Linux environments.

## Interpretation boundary

The deployment supports only the claim that the frozen evidence synthesis is portable and reproducible under the recorded contract. It does not repair the failed Week 2 Judge calibration, make Week 3–5 diagnostic quality scores human-validated, establish causal effects, or demonstrate production PIC readiness. Those limitations remain explicitly carried into the methodology report and claim-evidence matrix.
