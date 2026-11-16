# Frozen routing analysis

validation (used for selection; optimistic)

Items: 200; bootstrap replicates: 200.

| Arm | Accuracy | Tokens |
|---|---:|---:|
| qwen3:0.6b\|S0 | 0.8000 | 125.8750 |
| qwen3:0.6b\|S1 | 0.8000 | 210.8850 |
| qwen3:0.6b\|S3 | 0.8650 | 310.7950 |
| qwen3:0.6b\|S4 | 0.8750 | 310.3200 |
| qwen3:1.7b\|S0 | 0.8700 | 142.1000 |
| qwen3:1.7b\|S1 | 0.8750 | 225.9500 |
| qwen3:1.7b\|S3 | 0.9150 | 325.8550 |
| qwen3:1.7b\|S4 | 0.9250 | 325.6000 |
| qwen3:4b\|S0 | 0.9600 | 157.6700 |
| qwen3:4b\|S1 | 0.9250 | 242.7650 |
| qwen3:4b\|S3 | 0.9500 | 341.6350 |
| qwen3:4b\|S4 | 0.9550 | 341.7450 |
| qwen3:8b\|S0 | 0.9450 | 173.5750 |
| qwen3:8b\|S1 | 0.9400 | 258.3900 |
| qwen3:8b\|S3 | 0.9650 | 358.4350 |
| qwen3:8b\|S4 | 0.9800 | 357.7800 |

| Family / learner / mode | Raw AUROC | Cross-fit AUROC | Calibration |
|---|---:|---:|---|
| F1/logreg/per_arm | 0.5377 | 0.5377 | none |
| F1/logreg/shared | 0.5278 | 0.5278 | none |
| F1/hgb/per_arm | 0.5412 | 0.3976 | platt |
| F1/hgb/shared | 0.5433 | 0.4161 | platt |
| F1/mlp/per_arm | 0.5601 | 0.4506 | platt |
| F1/mlp/shared | 0.5250 | 0.4336 | platt |
| F2/logreg/per_arm | 0.5276 | 0.4250 | platt |
| F2/logreg/shared | 0.5078 | 0.4589 | platt |
| F2/hgb/per_arm | 0.5334 | 0.4519 | platt |
| F2/hgb/shared | 0.5513 | 0.4298 | platt |
| F2/mlp/per_arm | 0.5273 | 0.4216 | platt |
| F2/mlp/shared | 0.5295 | 0.4245 | platt |

| Policy | AUC | Accuracy | Tokens | Probe tokens | Accuracy 95% CI |
|---|---:|---:|---:|---:|---|
| target/best-0.00 | 0.8903 | 0.9800 | 357.7800 | 0.0000 | [0.9600000000000004, 0.9951250000000005] |
| target/best-0.02 | 0.9491 | 0.9600 | 157.6700 | 0.0000 | [0.9250000000000004, 0.9900000000000002] |
| target/best-0.05 | 0.9491 | 0.9600 | 157.6700 | 0.0000 | [0.9250000000000004, 0.9900000000000002] |
| heuristic | 0.9010 | 0.9750 | 354.6600 | 0.0000 | [0.9550000000000003, 0.9950000000000003] |
| threshold/0.5 | NA | 0.8000 | 125.8750 | 0.0000 | [0.7398750000000002, 0.8650000000000002] |
| threshold/sweep | 0.9533 | 0.9800 | 334.8000 | 0.0000 | [0.9600000000000004, 0.9950000000000004] |
| cascade/size_S0/logprob | 0.9344 | 0.9950 | 273.4700 | 0.0000 | [0.9850000000000003, 1.0000000000000004] |
| cascade/size_S0/agreement | 0.9422 | 0.9950 | 259.8650 | 0.0000 | [0.9850000000000003, 1.0000000000000004] |
| cascade/size_S0/learned_F4 | 0.9673 | 0.9800 | 158.7100 | 0.0000 | [0.9548750000000003, 0.9950000000000004] |
| cascade/size_S0/learned_F4_F1 | 0.9712 | 0.9850 | 160.6500 | 0.0000 | [0.9600000000000004, 1.0000000000000002] |
| cascade/fixed_frontier/logprob | 0.9249 | 0.9950 | 296.8250 | 0.0000 | [0.9850000000000003, 1.0000000000000004] |
| cascade/fixed_frontier/agreement | 0.9397 | 0.9950 | 276.8000 | 0.0000 | [0.9850000000000003, 1.0000000000000004] |
| cascade/fixed_frontier/learned_F4 | 0.9630 | 0.9800 | 783.4250 | 0.0000 | [0.9600000000000004, 0.9951250000000005] |
| cascade/fixed_frontier/learned_F4_F1 | 0.9662 | 0.9800 | 161.4350 | 0.0000 | [0.9600000000000003, 0.9950000000000004] |
| cascade/qwen3:0.6b/logprob | 0.8946 | 0.9400 | 280.9400 | 0.0000 | [0.9050000000000002, 0.9651250000000005] |
| cascade/qwen3:0.6b/agreement | 0.8946 | 0.9400 | 280.9400 | 0.0000 | [0.9050000000000002, 0.9651250000000005] |
| cascade/qwen3:0.6b/learned_F4 | 0.9330 | 0.9550 | 191.7750 | 0.0000 | [0.9150000000000003, 0.9800000000000003] |
| cascade/qwen3:0.6b/learned_F4_F1 | 0.9334 | 0.9550 | 190.8400 | 0.0000 | [0.9198750000000001, 0.9800000000000003] |
| cascade/qwen3:1.7b/logprob | 0.9131 | 0.9450 | 309.4400 | 0.0000 | [0.9148750000000002, 0.9750000000000004] |
| cascade/qwen3:1.7b/agreement | 0.9131 | 0.9450 | 309.4400 | 0.0000 | [0.9148750000000002, 0.9750000000000004] |
| cascade/qwen3:1.7b/learned_F4 | 0.9350 | 0.9500 | 182.4750 | 0.0000 | [0.9200000000000003, 0.9800000000000004] |
| cascade/qwen3:1.7b/learned_F4_F1 | 0.9272 | 0.9400 | 178.6800 | 0.0000 | [0.9050000000000002, 0.9701250000000005] |
| cascade/qwen3:4b/logprob | 0.9534 | 0.9650 | 159.2650 | 0.0000 | [0.9350000000000003, 0.9900000000000004] |
| cascade/qwen3:4b/agreement | 0.9534 | 0.9650 | 159.2650 | 0.0000 | [0.9350000000000003, 0.9900000000000004] |
| cascade/qwen3:4b/learned_F4 | 0.9701 | 0.9850 | 168.2950 | 0.0000 | [0.9650000000000003, 1.0000000000000004] |
| cascade/qwen3:4b/learned_F4_F1 | 0.9697 | 0.9850 | 174.8800 | 0.0000 | [0.9650000000000003, 1.0000000000000004] |
| cascade/qwen3:8b/logprob | 0.9460 | 0.9800 | 356.1500 | 0.0000 | [0.9648750000000003, 0.9951250000000005] |
| cascade/qwen3:8b/agreement | 0.9460 | 0.9800 | 356.1500 | 0.0000 | [0.9648750000000003, 0.9951250000000005] |
| cascade/qwen3:8b/learned_F4 | 0.9457 | 0.9650 | 182.0600 | 0.0000 | [0.9400000000000004, 0.9850000000000003] |
| cascade/qwen3:8b/learned_F4_F1 | 0.9493 | 0.9700 | 186.2150 | 0.0000 | [0.9450000000000004, 0.9900000000000003] |
| selected/F1 | 0.9497 | 0.9850 | 325.8750 | 0.0000 | [0.9700000000000003, 1.0000000000000004] |
| selected/F2 | 0.9551 | 0.9850 | 335.4050 | 0.0000 | [0.9700000000000003, 1.0000000000000004] |
| selected/overall | 0.9551 | 0.9850 | 335.4050 | 0.0000 | [0.9700000000000003, 1.0000000000000004] |

| Family | Train sizes | Mean AUROC | Mean AUC |
|---|---|---|---|
| F1 | 50, 100, 200, 400, 600 | 0.5126, 0.5496, 0.5370, 0.5411, 0.5377 | 0.9141, 0.9325, 0.9291, 0.9401, 0.9480 |
| F2 | 50, 100, 200, 400, 600 | 0.5073, 0.5512, 0.5268, 0.5049, 0.5276 | 0.9311, 0.8960, 0.9220, 0.8898, 0.9108 |

S2 subset (separate analysis):
- n=100; best-single accuracy=0.9900; main+S2 routing AUC=0.9631.

Caveats:
- Synthetic labels are fixtures, not real capability evidence.
- Validation selects layers, calibration, policies, and operating points; cross-fitting does not remove selection optimism.
- Costs are replayed tokens, not live latency, FLOPs, or energy. Every F3 point includes probe overhead; the fixed reference anchor does not.
- HF/Ollama quantization differs. Missing adapter S0 rows use WS2 tokenized S0 probe lengths.
- Bootstrap conditions on fitted models and frozen choices; no refitting or reselection.
- Full per-arm metrics, paired CIs, McNemar tests, grids and provenance: report.json.
- F3-own and F3-small skipped: no existing features directory supplied
