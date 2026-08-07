# Week 4 Multimodal Robustness Results

> Public-image proxy; AI-assisted rubric scores are diagnostic.

| Condition | n parsed | Mean total /5 | Scene /2 | Decision /2 | Uncertainty /1 | Acceptable decision | Forbidden claim |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 20/20 | 4.900 | 1.950 | 1.950 | 1.000 | 0.950 | 0.000 |
| gaussian_noise_std_0.08 | 20/20 | 4.800 | 1.900 | 1.900 | 1.000 | 0.900 | 0.050 |
| brightness_0.60 | 20/20 | 4.750 | 1.850 | 1.900 | 1.000 | 0.900 | 0.000 |

| Perturbation | Mean score drop | Decision consistency | Eligible scenarios |
|---|---:|---:|---:|
| gaussian_noise_std_0.08 | 0.100 | 0.950 | 20 |
| brightness_0.60 | 0.150 | 0.950 | 20 |

