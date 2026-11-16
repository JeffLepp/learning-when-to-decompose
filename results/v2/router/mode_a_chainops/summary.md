# Frozen routing analysis

validation (used for selection; optimistic)

Items: 200; bootstrap replicates: 200.

| Arm | Accuracy | Tokens |
|---|---:|---:|
| qwen3:0.6b&#124;S0 | 0.5800 | 211.0950 |
| qwen3:0.6b&#124;S1 | 0.6300 | 295.2100 |
| qwen3:0.6b&#124;S3 | 0.6450 | 365.3150 |
| qwen3:0.6b&#124;S4 | 0.5950 | 365.2650 |
| qwen3:1.7b&#124;S0 | 0.7550 | 227.5750 |
| qwen3:1.7b&#124;S1 | 0.7400 | 312.6750 |
| qwen3:1.7b&#124;S3 | 0.7550 | 381.5450 |
| qwen3:1.7b&#124;S4 | 0.7350 | 380.4750 |
| qwen3:4b&#124;S0 | 0.8300 | 243.5850 |
| qwen3:4b&#124;S1 | 0.8150 | 328.1250 |
| qwen3:4b&#124;S3 | 0.8450 | 396.8650 |
| qwen3:4b&#124;S4 | 0.8000 | 397.0700 |
| qwen3:8b&#124;S0 | 0.8650 | 258.8750 |
| qwen3:8b&#124;S1 | 0.8450 | 343.5600 |
| qwen3:8b&#124;S3 | 0.8600 | 412.8600 |
| qwen3:8b&#124;S4 | 0.9100 | 412.7850 |

| Family / learner / mode | Raw AUROC | Cross-fit AUROC | Calibration |
|---|---:|---:|---|
| F1/logreg/per_arm | 0.6631 | 0.6300 | platt |
| F1/logreg/shared | 0.6945 | 0.6683 | platt |
| F1/hgb/per_arm | 0.6237 | 0.5903 | platt |
| F1/hgb/shared | 0.6731 | 0.6414 | platt |
| F1/mlp/per_arm | 0.5560 | 0.4827 | platt |
| F1/mlp/shared | 0.6136 | 0.5500 | platt |
| F2/logreg/per_arm | 0.5842 | 0.5279 | platt |
| F2/logreg/shared | 0.5923 | 0.5323 | platt |
| F2/hgb/per_arm | 0.6148 | 0.5684 | platt |
| F2/hgb/shared | 0.6641 | 0.6252 | platt |
| F2/mlp/per_arm | 0.5561 | 0.4820 | platt |
| F2/mlp/shared | 0.5754 | 0.5246 | platt |

| Policy | AUC | Accuracy | Tokens | Probe tokens | Accuracy 95% CI |
|---|---:|---:|---:|---:|---|
| target/best-0.00 | 0.7451 | 0.9100 | 412.7850 | 0.0000 | [0.8700000000000003, 0.9400000000000004] |
| target/best-0.02 | 0.7451 | 0.9100 | 412.7850 | 0.0000 | [0.8700000000000003, 0.9400000000000004] |
| target/best-0.05 | 0.8313 | 0.8650 | 258.8750 | 0.0000 | [0.8250000000000002, 0.9051250000000004] |
| heuristic | 0.8025 | 0.9100 | 387.1150 | 0.0000 | [0.8700000000000003, 0.9450000000000003] |
| threshold/0.5 | 0.6830 | 0.6850 | 218.6100 | 0.0000 | [0.6100000000000001, 0.7401250000000003] |
| threshold/sweep | 0.8655 | 0.9250 | 328.8950 | 0.0000 | [0.8850000000000002, 0.9650000000000003] |
| cascade/size_S0/logprob | 0.7218 | 0.9650 | 489.5900 | 0.0000 | [0.9400000000000003, 0.9900000000000002] |
| cascade/size_S0/agreement | 0.7280 | 0.9450 | 548.3900 | 0.0000 | [0.9100000000000004, 0.9750000000000002] |
| cascade/size_S0/learned_F4 | 0.7744 | 0.9350 | 393.6850 | 0.0000 | [0.9000000000000004, 0.9700000000000004] |
| cascade/size_S0/learned_F4_F1 | 0.7735 | 0.9300 | 391.5150 | 0.0000 | [0.8900000000000003, 0.9650000000000003] |
| cascade/fixed_frontier/logprob | 0.7095 | 0.9900 | 537.1400 | 0.0000 | [0.9750000000000003, 1.0000000000000004] |
| cascade/fixed_frontier/agreement | 0.7271 | 0.9650 | 616.0700 | 0.0000 | [0.9398750000000003, 0.9850000000000003] |
| cascade/fixed_frontier/learned_F4 | 0.7760 | 0.9500 | 400.8950 | 0.0000 | [0.9200000000000004, 0.9800000000000003] |
| cascade/fixed_frontier/learned_F4_F1 | 0.7708 | 0.9450 | 403.6400 | 0.0000 | [0.9098750000000002, 0.9800000000000002] |
| cascade/qwen3:0.6b/logprob | 0.6808 | 0.7800 | 415.2250 | 0.0000 | [0.7250000000000002, 0.8400000000000002] |
| cascade/qwen3:0.6b/agreement | 0.6808 | 0.7800 | 415.2250 | 0.0000 | [0.7250000000000002, 0.8400000000000002] |
| cascade/qwen3:0.6b/learned_F4 | 0.6862 | 0.7750 | 394.9450 | 0.0000 | [0.7198750000000002, 0.8301250000000003] |
| cascade/qwen3:0.6b/learned_F4_F1 | 0.6813 | 0.7650 | 393.6950 | 0.0000 | [0.7000000000000002, 0.8152500000000004] |
| cascade/qwen3:1.7b/logprob | 0.8042 | 0.8800 | 429.9050 | 0.0000 | [0.8350000000000003, 0.9200000000000004] |
| cascade/qwen3:1.7b/agreement | 0.8042 | 0.8800 | 429.9050 | 0.0000 | [0.8350000000000003, 0.9200000000000004] |
| cascade/qwen3:1.7b/learned_F4 | 0.8258 | 0.8750 | 336.0000 | 0.0000 | [0.8300000000000003, 0.9200000000000003] |
| cascade/qwen3:1.7b/learned_F4_F1 | 0.8257 | 0.8750 | 336.5000 | 0.0000 | [0.8298750000000001, 0.9201250000000003] |
| cascade/qwen3:4b/logprob | 0.8462 | 0.9300 | 449.3250 | 0.0000 | [0.8948750000000002, 0.9650000000000003] |
| cascade/qwen3:4b/agreement | 0.8462 | 0.9300 | 449.3250 | 0.0000 | [0.8948750000000002, 0.9650000000000003] |
| cascade/qwen3:4b/learned_F4 | 0.8871 | 0.9550 | 332.8900 | 0.0000 | [0.9300000000000002, 0.9800000000000004] |
| cascade/qwen3:4b/learned_F4_F1 | 0.8814 | 0.9450 | 331.0100 | 0.0000 | [0.9100000000000003, 0.9751250000000005] |
| cascade/qwen3:8b/logprob | 0.8584 | 0.9550 | 468.8550 | 0.0000 | [0.9250000000000003, 0.9800000000000004] |
| cascade/qwen3:8b/agreement | 0.8584 | 0.9550 | 468.8550 | 0.0000 | [0.9250000000000003, 0.9800000000000004] |
| cascade/qwen3:8b/learned_F4 | 0.8869 | 0.9550 | 317.4450 | 0.0000 | [0.9300000000000004, 0.9850000000000003] |
| cascade/qwen3:8b/learned_F4_F1 | 0.8861 | 0.9550 | 320.8650 | 0.0000 | [0.9300000000000003, 0.9800000000000003] |
| selected/F1 | 0.8647 | 0.9300 | 379.4400 | 0.0000 | [0.8900000000000003, 0.9601250000000003] |
| selected/F2 | 0.8716 | 0.9200 | 370.0650 | 0.0000 | [0.8748750000000003, 0.9550000000000003] |
| selected/overall | 0.8716 | 0.9200 | 370.0650 | 0.0000 | [0.8748750000000003, 0.9550000000000003] |

| Family | Train sizes | Mean AUROC | Mean AUC |
|---|---|---|---|
| F1 | 50, 100, 200, 400, 600 | 0.5747, 0.6081, 0.6226, 0.6526, 0.6631 | 0.8184, 0.8075, 0.7963, 0.8028, 0.8280 |
| F2 | 50, 100, 200, 400, 600 | 0.5586, 0.5800, 0.5844, 0.5889, 0.5842 | 0.7975, 0.8199, 0.8233, 0.8141, 0.8063 |

S2 subset (separate analysis):
- n=100; best-single accuracy=0.9300; main+S2 routing AUC=0.8473.

Caveats:
- Synthetic labels are fixtures, not real capability evidence.
- Validation selects layers, calibration, policies, and operating points; cross-fitting does not remove selection optimism.
- Costs are replayed tokens, not live latency, FLOPs, or energy. Every F3 point includes probe overhead; the fixed reference anchor does not.
- HF/Ollama quantization differs. Missing adapter S0 rows use WS2 tokenized S0 probe lengths.
- Bootstrap conditions on fitted models and frozen choices; no refitting or reselection.
- Full per-arm metrics, paired CIs, McNemar tests, grids and provenance: report.json.
- F3-own and F3-small skipped: no existing features directory supplied
