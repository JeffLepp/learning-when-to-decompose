# Frozen routing analysis

ood_length: frozen evaluation; no refitting or selection

Items: 300; bootstrap replicates: 10000.

| Arm | Accuracy | Tokens |
|---|---:|---:|
| qwen3:0.6b&#124;S0 | 0.0867 | 223.2467 |
| qwen3:0.6b&#124;S1 | 0.0967 | 308.6300 |
| qwen3:0.6b&#124;S3 | 0.8733 | 831.8700 |
| qwen3:0.6b&#124;S4 | 0.9700 | 831.8367 |
| qwen3:1.7b&#124;S0 | 0.1967 | 240.1333 |
| qwen3:1.7b&#124;S1 | 0.1100 | 324.3867 |
| qwen3:1.7b&#124;S3 | 0.9467 | 847.1567 |
| qwen3:1.7b&#124;S4 | 0.9700 | 847.7467 |
| qwen3:4b&#124;S0 | 0.2400 | 255.2200 |
| qwen3:4b&#124;S1 | 0.1567 | 340.2967 |
| qwen3:4b&#124;S3 | 0.9467 | 864.2700 |
| qwen3:4b&#124;S4 | 0.9767 | 863.3200 |
| qwen3:8b&#124;S0 | 0.3533 | 271.8900 |
| qwen3:8b&#124;S1 | 0.2333 | 356.3500 |
| qwen3:8b&#124;S3 | 0.9667 | 879.3000 |
| qwen3:8b&#124;S4 | 0.9933 | 879.2467 |

| Family / learner / mode | Raw AUROC | Frozen Brier | Frozen ECE |
|---|---:|---:|---:|
| F1/logreg/per_arm | 0.5875 | 0.0932 | 0.0534 |
| F1/logreg/shared | 0.5811 | 0.1021 | 0.0836 |
| F1/hgb/per_arm | 0.5068 | 0.2340 | 0.2797 |
| F1/hgb/shared | 0.5061 | 0.1565 | 0.1785 |
| F1/mlp/per_arm | 0.5185 | 0.2348 | 0.2686 |
| F1/mlp/shared | 0.5369 | 0.1759 | 0.2012 |
| F2/logreg/per_arm | 0.5087 | 0.2598 | 0.3071 |
| F2/logreg/shared | 0.5214 | 0.2148 | 0.2657 |
| F2/hgb/per_arm | 0.4990 | 0.2676 | 0.3121 |
| F2/hgb/shared | 0.5213 | 0.2386 | 0.2856 |
| F2/mlp/per_arm | 0.5270 | 0.2872 | 0.3288 |
| F2/mlp/shared | 0.5045 | 0.2896 | 0.3310 |
| F3-own/logreg/per_arm | 0.6108 | 0.1261 | 0.1201 |
| F3-own/logreg/shared | 0.6119 | 0.1064 | 0.1109 |
| F3-own/hgb/per_arm | 0.5010 | 0.2024 | 0.2377 |
| F3-own/hgb/shared | 0.4925 | 0.1616 | 0.1867 |
| F3-own/mlp/per_arm | 0.5507 | 0.2216 | 0.2624 |
| F3-own/mlp/shared | 0.5376 | 0.2458 | 0.2930 |
| F3-small/logreg/per_arm | 0.6280 | 0.1180 | 0.1085 |
| F3-small/logreg/shared | 0.6380 | 0.1116 | 0.1148 |
| F3-small/hgb/per_arm | 0.4844 | 0.2035 | 0.2377 |
| F3-small/hgb/shared | 0.5239 | 0.1820 | 0.2247 |
| F3-small/mlp/per_arm | 0.6169 | 0.2127 | 0.2578 |
| F3-small/mlp/shared | 0.5277 | 0.2408 | 0.2834 |

H3 mean raw AUROC differences [95% paired CI]; columns are left minus right.
| Learner / mode | F3-own - F2 | F3-own - F1 | F2 - F1 | F3-own - F3-small | F3-small - F2 |
|---|---|---|---|---|---|
| logreg/per_arm | 0.1020 [0.0350, 0.1684] | 0.0233 [-0.0216, 0.0675] | -0.0787 [-0.1236, -0.0318] | -0.0173 [-0.0660, 0.0351] | 0.1193 [0.0654, 0.1676] |
| logreg/shared | 0.0905 [0.0461, 0.1326] | 0.0308 [0.0088, 0.0538] | -0.0597 [-0.1002, -0.0177] | -0.0261 [-0.0475, -0.0033] | 0.1166 [0.0721, 0.1579] |
| hgb/per_arm | 0.0020 [-0.0580, 0.0631] | -0.0058 [-0.0632, 0.0504] | -0.0077 [-0.0697, 0.0522] | 0.0166 [-0.0294, 0.0663] | -0.0146 [-0.0830, 0.0534] |
| hgb/shared | -0.0288 [-0.0839, 0.0285] | -0.0137 [-0.0679, 0.0410] | 0.0151 [-0.0457, 0.0741] | -0.0314 [-0.0748, 0.0275] | 0.0026 [-0.0609, 0.0607] |
| mlp/per_arm | 0.0237 [-0.0350, 0.0813] | 0.0321 [-0.0202, 0.0834] | 0.0085 [-0.0384, 0.0562] | -0.0663 [-0.1164, -0.0083] | 0.0899 [0.0323, 0.1358] |
| mlp/shared | 0.0331 [-0.0175, 0.0807] | 0.0007 [-0.0411, 0.0452] | -0.0323 [-0.0777, 0.0219] | 0.0099 [-0.0447, 0.0672] | 0.0232 [-0.0393, 0.0769] |
Single-class arm replicates are skipped and counted per predictor in JSON; absent-family reasons are logged there.

| H4/H5 contrast | Normalized AUC difference [95% CI] |
|---|---|
| selected/overall - best_single | 0.0846 [0.0525, 0.1165] |
| selected/overall - fixed_frontier | -0.0245 [-0.0458, 0.0409] |
| selected/cascade - selected/overall | -0.2576 [-0.3043, -0.2093] |
| selected/F1 - selected/F1 | 0.0000 [0.0000, 0.0000] |
| selected/F2 - selected/F1 | 0.0012 [-0.0276, 0.0318] |
| selected/F3-own - selected/F1 | -0.5112 [-0.5466, -0.4194] |
| selected/F3-small - selected/F1 | -0.1194 [-0.1470, -0.0920] |
H5 D1.7: accuracy delta -0.1200 [-0.1633, -0.0767]; tokens delta 264.8100 [179.8480, 352.7586]; McNemar {'b': 6, 'c': 42, 'p_exact': 1.0087482138487759e-07}.

| Policy | AUC | Accuracy | Tokens | Probe tokens | Accuracy 95% CI |
|---|---:|---:|---:|---:|---|
| target/best-0.00 | 0.5425 | 0.9767 | 863.3200 | 0.0000 | [0.9599999999999991, 0.9933333333333325] |
| target/best-0.02 | 0.5425 | 0.9767 | 863.3200 | 0.0000 | [0.9599999999999991, 0.9933333333333325] |
| target/best-0.05 | 0.5084 | 0.8733 | 831.8700 | 0.0000 | [0.8333333333333328, 0.9099999999999995] |
| heuristic | 0.5609 | 0.9933 | 879.2467 | 0.0000 | [0.9833333333333326, 0.9999999999999994] |
| threshold/0.5 | 0.5655 | 0.8100 | 666.7367 | 0.0000 | [0.7633333333333329, 0.8533333333333328] |
| threshold/sweep | 0.5964 | 0.9867 | 865.6167 | 0.0000 | [0.9733333333333326, 0.9966666666666661] |
| cascade/size_S0/logprob | 0.3556 | 0.4967 | 688.5967 | 0.0000 | [0.4399999999999999, 0.553333333333333] |
| cascade/size_S0/agreement | 0.2609 | 0.3700 | 820.3233 | 0.0000 | [0.3166666666666666, 0.42333333333333323] |
| cascade/size_S0/learned_F4 | 0.3481 | 0.5200 | 775.1100 | 0.0000 | [0.46333333333333315, 0.5766666666666664] |
| cascade/size_S0/learned_F4_F1 | 0.3532 | 0.5133 | 790.0600 | 0.0000 | [0.4566666666666665, 0.5699999999999998] |
| cascade/fixed_frontier/logprob | 0.3695 | 0.8600 | 1118.1433 | 0.0000 | [0.8199999999999994, 0.8999999999999992] |
| cascade/fixed_frontier/agreement | 0.2635 | 0.8467 | 1913.7733 | 0.0000 | [0.803333333333333, 0.886666666666666] |
| cascade/fixed_frontier/learned_F4 | 0.3697 | 0.8067 | 1088.6433 | 0.0000 | [0.7599999999999996, 0.8499999999999994] |
| cascade/fixed_frontier/learned_F4_F1 | 0.3707 | 0.7933 | 1078.9433 | 0.0000 | [0.7466666666666661, 0.8366666666666662] |
| cascade/qwen3:0.6b/logprob | 0.4243 | 0.8567 | 962.0933 | 0.0000 | [0.8166666666666661, 0.8966666666666661] |
| cascade/qwen3:0.6b/agreement | 0.4243 | 0.8567 | 962.0933 | 0.0000 | [0.8166666666666661, 0.8966666666666661] |
| cascade/qwen3:0.6b/learned_F4 | 0.4255 | 0.8767 | 1012.6933 | 0.0000 | [0.8366666666666662, 0.9133333333333327] |
| cascade/qwen3:0.6b/learned_F4_F1 | 0.4267 | 0.8800 | 1009.8133 | 0.0000 | [0.8433333333333327, 0.9166666666666659] |
| cascade/qwen3:1.7b/logprob | 0.5402 | 0.9400 | 936.3333 | 0.0000 | [0.9133333333333326, 0.9666666666666659] |
| cascade/qwen3:1.7b/agreement | 0.5402 | 0.9400 | 936.3333 | 0.0000 | [0.9133333333333326, 0.9666666666666659] |
| cascade/qwen3:1.7b/learned_F4 | 0.5379 | 0.9500 | 999.0433 | 0.0000 | [0.9233333333333327, 0.9733333333333327] |
| cascade/qwen3:1.7b/learned_F4_F1 | 0.5378 | 0.9433 | 1000.6067 | 0.0000 | [0.916666666666666, 0.9666666666666661] |
| cascade/qwen3:4b/logprob | 0.5463 | 0.9133 | 915.8200 | 0.0000 | [0.8799999999999994, 0.9433333333333327] |
| cascade/qwen3:4b/agreement | 0.5463 | 0.9133 | 915.8200 | 0.0000 | [0.8799999999999994, 0.9433333333333327] |
| cascade/qwen3:4b/learned_F4 | 0.5452 | 0.9333 | 937.7033 | 0.0000 | [0.9033333333333328, 0.9599999999999993] |
| cascade/qwen3:4b/learned_F4_F1 | 0.5463 | 0.9367 | 940.4433 | 0.0000 | [0.9066666666666661, 0.9633333333333327] |
| cascade/qwen3:8b/logprob | 0.6314 | 0.8367 | 739.8433 | 0.0000 | [0.7933333333333328, 0.8766666666666662] |
| cascade/qwen3:8b/agreement | 0.6314 | 0.8367 | 739.8433 | 0.0000 | [0.7933333333333328, 0.8766666666666662] |
| cascade/qwen3:8b/learned_F4 | 0.6306 | 0.8867 | 790.2700 | 0.0000 | [0.8499999999999993, 0.9199999999999995] |
| cascade/qwen3:8b/learned_F4_F1 | 0.6204 | 0.9167 | 837.9700 | 0.0000 | [0.8833333333333327, 0.9466666666666661] |
| selected/F1 | 0.6271 | 0.9800 | 853.3333 | 0.0000 | [0.9633333333333326, 0.9933333333333327] |
| selected/F2 | 0.6284 | 0.9667 | 858.3367 | 0.0000 | [0.9433333333333329, 0.9866666666666659] |
| selected/F3-own | 0.1160 | 0.9800 | 1431.1900 | 568.7200 | [0.9633333333333326, 0.9933333333333327] |
| selected/F3-small | 0.5077 | 0.9833 | 991.6900 | 118.1800 | [0.966666666666666, 0.996666666666666] |
| selected/overall | 0.6271 | 0.9800 | 853.3333 | 0.0000 | [0.9633333333333326, 0.9933333333333327] |
| selected/cascade | 0.3695 | 0.8600 | 1118.1433 | 0.0000 | [0.8199999999999994, 0.8999999999999992] |

S2 subset (separate analysis):
- n=100; best-single accuracy=1.0000; main+S2 routing AUC=0.6306.

Caveats:
- Synthetic labels are fixtures, not real capability evidence.
- Validation selects layers, calibration, policies, and operating points; cross-fitting does not remove selection optimism.
- Costs are replayed tokens, not live latency, FLOPs, or energy. Every F3 point includes probe overhead; the fixed reference anchor does not.
- HF/Ollama quantization differs. Missing adapter S0 rows use WS2 tokenized S0 probe lengths.
- Bootstrap conditions on fitted models and frozen choices; no refitting or reselection.
- Full per-arm metrics, paired CIs, McNemar tests, grids and provenance: report.json.
