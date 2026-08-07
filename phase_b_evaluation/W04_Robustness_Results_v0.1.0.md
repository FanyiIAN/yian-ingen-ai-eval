# Week 4 Text Robustness Results

> AI-assisted rubric scores are diagnostic and not human ground truth.

| Model | Parsed | Semantic robustness | Stable pass | Stable fail | Task degradation at 60% mask | Mandatory review flags |
|---|---:|---:|---:|---:|---:|---:|
| flan_t5_base | 182/182 | 0.914 | 7 | 25 | 0.357 | 5 |
| llama31_8b_instruct | 182/182 | 0.857 | 26 | 4 | 0.286 | 12 |
| mistral_7b_instruct_v0_2 | 182/182 | 0.857 | 30 | 0 | 0.000 | 9 |

## Masked-input curves

### flan_t5_base

| Mask ratio | Mean task accuracy | Severity-weighted mean | Pass rate | Original flip rate | Severity-5 failures |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 3.643 | 4.119 | 0.429 | N/A | 0 |
| 0.2 | 3.500 | 3.976 | 0.429 | 0.000 | 0 |
| 0.4 | 3.286 | 3.762 | 0.286 | 0.143 | 0 |
| 0.6 | 3.286 | 3.762 | 0.286 | 0.143 | 0 |

### llama31_8b_instruct

| Mask ratio | Mean task accuracy | Severity-weighted mean | Pass rate | Original flip rate | Severity-5 failures |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 4.643 | 4.881 | 0.929 | N/A | 0 |
| 0.2 | 4.500 | 4.690 | 0.786 | 0.143 | 0 |
| 0.4 | 4.571 | 4.762 | 0.786 | 0.143 | 0 |
| 0.6 | 4.357 | 4.643 | 0.714 | 0.214 | 0 |

### mistral_7b_instruct_v0_2

| Mask ratio | Mean task accuracy | Severity-weighted mean | Pass rate | Original flip rate | Severity-5 failures |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 4.071 | 4.214 | 0.929 | N/A | 0 |
| 0.2 | 4.071 | 4.167 | 0.929 | 0.000 | 0 |
| 0.4 | 3.929 | 4.024 | 0.929 | 0.000 | 0 |
| 0.6 | 4.071 | 4.214 | 0.929 | 0.000 | 0 |

