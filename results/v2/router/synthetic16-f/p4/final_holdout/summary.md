# Frozen routing analysis

holdout: frozen evaluation; no refitting or selection

Items: 400; bootstrap replicates: 10000.

| Arm | Accuracy | Tokens |
|---|---:|---:|
| qwen3:0.6b&#124;S0 | 0.7225 | 135.2450 |
| qwen3:0.6b&#124;S1 | 0.6975 | 219.6000 |
| qwen3:0.6b&#124;S3 | 0.8400 | 362.9425 |
| qwen3:0.6b&#124;S4 | 0.8850 | 362.8725 |
| qwen3:1.7b&#124;S0 | 0.8050 | 150.5150 |
| qwen3:1.7b&#124;S1 | 0.7700 | 236.0175 |
| qwen3:1.7b&#124;S3 | 0.9175 | 378.8850 |
| qwen3:1.7b&#124;S4 | 0.9200 | 378.6100 |
| qwen3:4b&#124;S0 | 0.8850 | 166.6150 |
| qwen3:4b&#124;S1 | 0.8475 | 251.8500 |
| qwen3:4b&#124;S3 | 0.9275 | 394.9275 |
| qwen3:4b&#124;S4 | 0.9475 | 394.6875 |
| qwen3:8b&#124;S0 | 0.9125 | 183.0325 |
| qwen3:8b&#124;S1 | 0.8925 | 268.6675 |
| qwen3:8b&#124;S3 | 0.9450 | 410.7875 |
| qwen3:8b&#124;S4 | 0.9575 | 410.7875 |

| Family / learner / mode | Raw AUROC | Frozen Brier | Frozen ECE |
|---|---:|---:|---:|
| F1/logreg/per_arm | 0.6275 | 0.1029 | 0.0286 |
| F1/logreg/shared | 0.5909 | 0.1026 | 0.0336 |
| F1/hgb/per_arm | 0.5845 | 0.1070 | 0.0299 |
| F1/hgb/shared | 0.6197 | 0.1047 | 0.0306 |
| F1/mlp/per_arm | 0.5706 | 0.1078 | 0.0285 |
| F1/mlp/shared | 0.5897 | 0.1057 | 0.0392 |
| F2/logreg/per_arm | 0.5524 | 0.1081 | 0.0319 |
| F2/logreg/shared | 0.5671 | 0.1058 | 0.0302 |
| F2/hgb/per_arm | 0.5632 | 0.1085 | 0.0285 |
| F2/hgb/shared | 0.5841 | 0.1057 | 0.0318 |
| F2/mlp/per_arm | 0.5325 | 0.1089 | 0.0235 |
| F2/mlp/shared | 0.5318 | 0.1093 | 0.0231 |
| F3-own/logreg/per_arm | 0.6288 | 0.1036 | 0.0325 |
| F3-own/logreg/shared | 0.6194 | 0.1032 | 0.0370 |
| F3-own/hgb/per_arm | 0.5898 | 0.1057 | 0.0321 |
| F3-own/hgb/shared | 0.6199 | 0.1047 | 0.0357 |
| F3-own/mlp/per_arm | 0.5794 | 0.1077 | 0.0301 |
| F3-own/mlp/shared | 0.5655 | 0.1081 | 0.0298 |
| F3-small/logreg/per_arm | 0.6314 | 0.1037 | 0.0346 |
| F3-small/logreg/shared | 0.6167 | 0.1026 | 0.0383 |
| F3-small/hgb/per_arm | 0.5951 | 0.1051 | 0.0303 |
| F3-small/hgb/shared | 0.6141 | 0.1033 | 0.0340 |
| F3-small/mlp/per_arm | 0.5749 | 0.1079 | 0.0284 |
| F3-small/mlp/shared | 0.5860 | 0.1075 | 0.0286 |

H3 mean raw AUROC differences [95% paired CI]; columns are left minus right.
| Learner / mode | F3-own - F2 | F3-own - F1 | F2 - F1 | F3-own - F3-small | F3-small - F2 |
|---|---|---|---|---|---|
| logreg/per_arm | 0.0764 [0.0401, 0.1120] | 0.0013 [-0.0248, 0.0280] | -0.0751 [-0.1032, -0.0464] | -0.0026 [-0.0295, 0.0253] | 0.0790 [0.0441, 0.1138] |
| logreg/shared | 0.0523 [0.0300, 0.0754] | 0.0284 [0.0129, 0.0446] | -0.0239 [-0.0415, -0.0065] | 0.0027 [-0.0103, 0.0162] | 0.0496 [0.0284, 0.0714] |
| hgb/per_arm | 0.0266 [-0.0045, 0.0573] | 0.0053 [-0.0239, 0.0350] | -0.0213 [-0.0456, 0.0036] | -0.0053 [-0.0371, 0.0264] | 0.0319 [0.0007, 0.0639] |
| hgb/shared | 0.0358 [0.0093, 0.0620] | 0.0002 [-0.0290, 0.0283] | -0.0356 [-0.0566, -0.0134] | 0.0058 [-0.0204, 0.0320] | 0.0300 [0.0011, 0.0592] |
| mlp/per_arm | 0.0470 [0.0113, 0.0827] | 0.0089 [-0.0224, 0.0394] | -0.0381 [-0.0660, -0.0092] | 0.0045 [-0.0286, 0.0380] | 0.0425 [0.0036, 0.0815] |
| mlp/shared | 0.0336 [-0.0005, 0.0669] | -0.0242 [-0.0549, 0.0068] | -0.0579 [-0.0871, -0.0275] | -0.0206 [-0.0520, 0.0109] | 0.0542 [0.0229, 0.0847] |
Single-class arm replicates are skipped and counted per predictor in JSON; absent-family reasons are logged there.

| H4/H5 contrast | Normalized AUC difference [95% CI] |
|---|---|
| selected/overall - best_single | 0.0739 [0.0490, 0.1011] |
| selected/overall - fixed_frontier | 0.0098 [-0.0111, 0.0223] |
| selected/cascade - selected/overall | 0.0478 [0.0280, 0.0648] |
| selected/F1 - selected/F1 | 0.0000 [0.0000, 0.0000] |
| selected/F2 - selected/F1 | -0.0108 [-0.0299, 0.0073] |
| selected/F3-own - selected/F1 | -0.1826 [-0.2243, -0.1432] |
| selected/F3-small - selected/F1 | -0.0487 [-0.0662, -0.0311] |
H5 D1.7: accuracy delta 0.0350 [0.0075, 0.0625]; tokens delta -152.4700 [-167.4013, -136.9547]; McNemar {'b': 23, 'c': 9, 'p_exact': 0.020061607006937265}.

| Policy | AUC | Accuracy | Tokens | Probe tokens | Accuracy 95% CI |
|---|---:|---:|---:|---:|---|
| target/best-0.00 | 0.8416 | 0.9475 | 394.6875 | 0.0000 | [0.9250000000000005, 0.9675000000000007] |
| target/best-0.02 | 0.8416 | 0.9475 | 394.6875 | 0.0000 | [0.9250000000000005, 0.9675000000000007] |
| target/best-0.05 | 0.7915 | 0.8400 | 362.9425 | 0.0000 | [0.8025000000000004, 0.8750000000000004] |
| heuristic | 0.8756 | 0.9500 | 354.8050 | 0.0000 | [0.9275000000000005, 0.9700000000000006] |
| threshold/0.5 | 0.7673 | 0.7675 | 137.8975 | 0.0000 | [0.7250000000000003, 0.8075000000000006] |
| threshold/sweep | 0.9089 | 0.9525 | 405.0300 | 0.0000 | [0.9300000000000006, 0.9725000000000006] |
| cascade/size_S0/logprob | 0.9587 | 0.9475 | 193.1375 | 0.0000 | [0.9250000000000005, 0.9675000000000006] |
| cascade/size_S0/agreement | 0.9309 | 0.9350 | 237.4550 | 0.0000 | [0.9100000000000006, 0.9575000000000006] |
| cascade/size_S0/learned_F4 | 0.9554 | 0.9525 | 193.9125 | 0.0000 | [0.9300000000000006, 0.9725000000000006] |
| cascade/size_S0/learned_F4_F1 | 0.9533 | 0.9650 | 201.4000 | 0.0000 | [0.9450000000000006, 0.9825000000000006] |
| cascade/fixed_frontier/logprob | 0.9633 | 0.9750 | 208.9350 | 0.0000 | [0.9575000000000007, 0.9875000000000007] |
| cascade/fixed_frontier/agreement | 0.9299 | 0.9675 | 302.1850 | 0.0000 | [0.9500000000000005, 0.9825000000000006] |
| cascade/fixed_frontier/learned_F4 | 0.9586 | 0.9550 | 195.3075 | 0.0000 | [0.9325000000000006, 0.9750000000000005] |
| cascade/fixed_frontier/learned_F4_F1 | 0.9567 | 0.9550 | 196.5725 | 0.0000 | [0.9325000000000006, 0.9750000000000005] |
| cascade/qwen3:0.6b/logprob | 0.8896 | 0.9300 | 275.2925 | 0.0000 | [0.9025000000000005, 0.9525000000000007] |
| cascade/qwen3:0.6b/agreement | 0.8896 | 0.9300 | 275.2925 | 0.0000 | [0.9025000000000005, 0.9525000000000007] |
| cascade/qwen3:0.6b/learned_F4 | 0.8901 | 0.9325 | 287.4375 | 0.0000 | [0.9075000000000005, 0.9550000000000006] |
| cascade/qwen3:0.6b/learned_F4_F1 | 0.8890 | 0.9300 | 267.0450 | 0.0000 | [0.9050000000000005, 0.9525000000000006] |
| cascade/qwen3:1.7b/logprob | 0.9281 | 0.9600 | 285.4525 | 0.0000 | [0.9400000000000005, 0.9775000000000006] |
| cascade/qwen3:1.7b/agreement | 0.9281 | 0.9600 | 285.4525 | 0.0000 | [0.9400000000000005, 0.9775000000000006] |
| cascade/qwen3:1.7b/learned_F4 | 0.9288 | 0.9600 | 281.4975 | 0.0000 | [0.9400000000000006, 0.9775000000000006] |
| cascade/qwen3:1.7b/learned_F4_F1 | 0.9272 | 0.9575 | 278.3475 | 0.0000 | [0.9375000000000004, 0.9750000000000006] |
| cascade/qwen3:4b/logprob | 0.9553 | 0.9875 | 243.8825 | 0.0000 | [0.9750000000000006, 0.9975000000000006] |
| cascade/qwen3:4b/agreement | 0.9553 | 0.9875 | 243.8825 | 0.0000 | [0.9750000000000006, 0.9975000000000006] |
| cascade/qwen3:4b/learned_F4 | 0.9581 | 0.9800 | 219.2575 | 0.0000 | [0.9650000000000005, 0.9925000000000006] |
| cascade/qwen3:4b/learned_F4_F1 | 0.9559 | 0.9775 | 221.5375 | 0.0000 | [0.9625000000000006, 0.9900000000000007] |
| cascade/qwen3:8b/logprob | 0.9562 | 0.9800 | 223.2725 | 0.0000 | [0.9650000000000005, 0.9925000000000006] |
| cascade/qwen3:8b/agreement | 0.9562 | 0.9800 | 223.2725 | 0.0000 | [0.9650000000000005, 0.9925000000000006] |
| cascade/qwen3:8b/learned_F4 | 0.9556 | 0.9525 | 204.3100 | 0.0000 | [0.9300000000000006, 0.9725000000000006] |
| cascade/qwen3:8b/learned_F4_F1 | 0.9540 | 0.9650 | 210.9700 | 0.0000 | [0.9450000000000006, 0.9825000000000006] |
| selected/F1 | 0.9155 | 0.9400 | 361.4050 | 0.0000 | [0.9150000000000006, 0.9625000000000006] |
| selected/F2 | 0.9047 | 0.9425 | 379.7050 | 0.0000 | [0.9200000000000005, 0.9650000000000005] |
| selected/F3-own | 0.7329 | 0.9450 | 812.5600 | 426.9800 | [0.9225000000000005, 0.9650000000000006] |
| selected/F3-small | 0.8668 | 0.9425 | 450.9875 | 82.7450 | [0.9199375000000004, 0.9650000000000005] |
| selected/overall | 0.9155 | 0.9400 | 361.4050 | 0.0000 | [0.9150000000000006, 0.9625000000000006] |
| selected/cascade | 0.9633 | 0.9750 | 208.9350 | 0.0000 | [0.9575000000000007, 0.9875000000000007] |

S2 subset (separate analysis):
- n=150; best-single accuracy=0.9400; main+S2 routing AUC=0.9105.

Caveats:
- Synthetic labels are fixtures, not real capability evidence.
- Validation selects layers, calibration, policies, and operating points; cross-fitting does not remove selection optimism.
- Costs are replayed tokens, not live latency, FLOPs, or energy. Every F3 point includes probe overhead; the fixed reference anchor does not.
- HF/Ollama quantization differs. Missing adapter S0 rows use WS2 tokenized S0 probe lengths.
- Bootstrap conditions on fitted models and frozen choices; no refitting or reselection.
- Full per-arm metrics, paired CIs, McNemar tests, grids and provenance: report.json.
