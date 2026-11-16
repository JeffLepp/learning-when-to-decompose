# Frozen routing analysis

validation (used for selection; optimistic)

Items: 32; bootstrap replicates: 10000.

| Arm | Accuracy | Tokens |
|---|---:|---:|
| qwen3:8b&#124;S0 | 0.0312 | 241.8125 |
| qwen3:8b&#124;S3 | 0.3125 | 688.0312 |

| Family / learner / mode | Raw AUROC | Cross-fit AUROC | Calibration | Platt fallback folds/full | Frozen Brier / ECE |
|---|---:|---:|---|---:|---|
| F1/logreg/per_arm | 0.8359 | 0.8359 | none | 0/0 | 0.1176 / 0.0902 |
| F1/logreg/shared | 0.6608 | 0.6608 | none | 0/0 | 0.1174 / 0.1204 |
| F1/hgb/per_arm | 0.7880 | 0.6880 | platt | 0/0 | 0.1166 / 0.0542 |
| F1/hgb/shared | 0.6637 | 0.3120 | platt | 0/0 | 0.1142 / 0.0471 |
| F1/mlp/per_arm | 0.7789 | 0.7359 | platt | 0/0 | 0.1112 / 0.0822 |
| F1/mlp/shared | 0.7218 | 0.6355 | platt | 0/0 | 0.1179 / 0.0459 |
| F2/logreg/per_arm | 0.8173 | 0.6937 | platt | 0/0 | 0.1050 / 0.0870 |
| F2/logreg/shared | 0.6517 | 0.3711 | platt | 1/0 | 0.1057 / 0.0319 |
| F2/hgb/per_arm | 0.7648 | 0.6923 | platt | 0/0 | 0.1163 / 0.0510 |
| F2/hgb/shared | 0.6496 | 0.3463 | platt | 0/0 | 0.1123 / 0.0207 |
| F2/mlp/per_arm | 0.8839 | 0.7555 | platt | 0/0 | 0.0909 / 0.0809 |
| F2/mlp/shared | 0.6953 | 0.3734 | platt | 1/0 | 0.1016 / 0.0887 |
| F3-own/logreg/per_arm | 0.8518 | 0.7930 | platt | 0/0 | 0.0956 / 0.0670 |
| F3-own/logreg/shared | 0.8562 | 0.7689 | platt | 0/0 | 0.0955 / 0.0836 |
| F3-own/hgb/per_arm | 0.7123 | 0.4326 | platt | 0/0 | 0.1172 / 0.0876 |
| F3-own/hgb/shared | 0.7837 | 0.7414 | isotonic | 0/0 | 0.0871 / 0.0000 |
| F3-own/mlp/per_arm | 0.8341 | 0.7636 | platt | 0/0 | 0.0968 / 0.0799 |
| F3-own/mlp/shared | 0.7902 | 0.7380 | platt | 0/0 | 0.1101 / 0.0537 |
| F3-small/logreg/per_arm | 0.8818 | 0.8341 | platt | 0/0 | 0.0863 / 0.0514 |
| F3-small/logreg/shared | 0.7580 | 0.6664 | platt | 0/0 | 0.1140 / 0.0581 |
| F3-small/hgb/per_arm | 0.8510 | 0.8135 | isotonic | 0/0 | 0.0540 / 0.0000 |
| F3-small/hgb/shared | 0.9295 | 0.9295 | none | 0/0 | 0.0808 / 0.0826 |
| F3-small/mlp/per_arm | 0.9045 | 0.8545 | platt | 0/0 | 0.0821 / 0.0454 |
| F3-small/mlp/shared | 0.7232 | 0.4570 | platt | 0/0 | 0.1107 / 0.0586 |

H3 mean raw AUROC differences [95% paired CI]; columns are left minus right.
| Learner / mode | F3-own - F2 | F3-own - F1 | F2 - F1 | F3-own - F3-small | F3-small - F2 |
|---|---|---|---|---|---|
| logreg/per_arm | 0.0345 [-0.1182, 0.1739] | 0.0159 [-0.1442, 0.2323] | -0.0186 [-0.1336, 0.1615] | -0.0300 [-0.1125, 0.1143] | 0.0645 [-0.1039, 0.1601] |
| logreg/shared | 0.2045 [-0.0273, 0.3235] | 0.1954 [-0.0526, 0.3159] | -0.0091 [-0.1771, 0.1441] | 0.0982 [0.0112, 0.2545] | 0.1063 [-0.1771, 0.2250] |
| hgb/per_arm | -0.0525 [-0.2296, 0.1167] | -0.0757 [-0.2065, 0.1082] | -0.0232 [-0.1150, 0.1198] | -0.1386 [-0.4227, -0.0426] | 0.0861 [-0.0149, 0.3550] |
| hgb/shared | 0.1340 [-0.1979, 0.2395] | 0.1199 [-0.1333, 0.2299] | -0.0141 [-0.1256, 0.1667] | -0.1459 [-0.4010, -0.0282] | 0.2799 [0.0400, 0.3982] |
| mlp/per_arm | -0.0498 [-0.2778, 0.0682] | 0.0552 [-0.1146, 0.1818] | 0.1050 [0.0027, 0.2914] | -0.0705 [-0.2597, 0.0208] | 0.0207 [-0.1154, 0.1547] |
| mlp/shared | 0.0949 [-0.1583, 0.1768] | 0.0684 [-0.0628, 0.2857] | -0.0265 [-0.1331, 0.3182] | 0.0670 [-0.1318, 0.1500] | 0.0279 [-0.1667, 0.1430] |
Single-class arm replicates are skipped and counted per predictor in JSON; absent-family reasons are logged there.

| H4/H5 contrast | Normalized AUC difference [95% CI] |
|---|---|
| selected/overall - best_single | 0.0568 [0.0014, 0.1133] |
| selected/overall - fixed_frontier | 0.0568 [0.0014, 0.1133] |
| selected/cascade - selected/overall | -0.1062 [-0.1658, -0.0438] |
| selected/F1 - selected/F1 | 0.0000 [0.0000, 0.0000] |
| selected/F2 - selected/F1 | -0.0066 [-0.0481, 0.0422] |
| selected/F3-own - selected/F1 | -0.1075 [-0.1732, -0.0449] |
| selected/F3-small - selected/F1 | -0.0872 [-0.1521, -0.0141] |
H5 D1.7: accuracy delta 0.0000 [-0.0938, 0.0938]; tokens delta 328.2188 [269.1555, 396.5008]; McNemar {'b': 1, 'c': 1, 'p_exact': 1.0}.

| Policy | AUC | Accuracy | Tokens | Probe tokens | Accuracy 95% CI |
|---|---:|---:|---:|---:|---|
| target/best-0.00 | 0.1719 | 0.3125 | 688.0312 | 0.0000 | [0.15625, 0.46875] |
| target/best-0.02 | 0.1719 | 0.3125 | 688.0312 | 0.0000 | [0.15625, 0.46875] |
| target/best-0.05 | 0.1719 | 0.3125 | 688.0312 | 0.0000 | [0.15625, 0.46875] |
| heuristic | 0.1663 | 0.3125 | 688.0312 | 0.0000 | [0.15625, 0.46875] |
| threshold/0.5 | 0.1902 | 0.3125 | 629.9375 | 0.0000 | [0.15625, 0.46875] |
| threshold/sweep | 0.1812 | 0.3125 | 629.9375 | 0.0000 | [0.15625, 0.46875] |
| cascade/fixed_frontier/agreement | 0.1225 | 0.3125 | 929.8438 | 0.0000 | [0.15625, 0.46875] |
| cascade/qwen3:8b/agreement | 0.1225 | 0.3125 | 929.8438 | 0.0000 | [0.15625, 0.46875] |
| selected/F1 | 0.2286 | 0.3125 | 601.6250 | 0.0000 | [0.15625, 0.46875] |
| selected/F2 | 0.2221 | 0.3125 | 673.5938 | 0.0000 | [0.15625, 0.46875] |
| selected/F3-own | 0.1211 | 0.3125 | 829.6250 | 185.5000 | [0.15625, 0.46875] |
| selected/F3-small | 0.1414 | 0.3125 | 602.5625 | 185.5000 | [0.15625, 0.46875] |
| selected/overall | 0.2286 | 0.3125 | 601.6250 | 0.0000 | [0.15625, 0.46875] |
| selected/cascade | 0.1225 | 0.3125 | 929.8438 | 0.0000 | [0.15625, 0.46875] |

| Family | Train sizes | Mean AUROC | Mean AUC |
|---|---|---|---|
| F1 | 50, 96 | 0.7625, 0.8359 | 0.2117, 0.2286 |
| F2 | 50, 96 | 0.7249, 0.8173 | 0.2219, 0.2440 |
| F3-own | 50, 96 | 0.7831, 0.8518 | 0.1224, 0.1395 |

S2 subset (separate analysis):
- No complete S2 subset available.

Caveats:
- Historical adapter results are exploratory pilot evidence.
- Validation selects layers, calibration, policies, and operating points; cross-fitting does not remove selection optimism.
- Costs are replayed tokens, not live latency, FLOPs, or energy. Every F3 point includes probe overhead; the fixed reference anchor does not.
- HF/Ollama quantization differs. Missing adapter S0 rows use WS2 tokenized S0 probe lengths.
- Bootstrap conditions on fitted models and frozen choices; no refitting or reselection.
- Full per-arm metrics, paired CIs, McNemar tests, grids and provenance: report.json.
- Cascade size_S0 skipped: fewer than two ladder arms
- cascade/fixed_frontier/logprob skipped: all ladder logprobs missing
- cascade/fixed_frontier/learned_F4 skipped: all ladder logprobs missing
- cascade/fixed_frontier/learned_F4_F1 skipped: all ladder logprobs missing
- cascade/qwen3:8b/logprob skipped: all ladder logprobs missing
- cascade/qwen3:8b/learned_F4 skipped: all ladder logprobs missing
- cascade/qwen3:8b/learned_F4_F1 skipped: all ladder logprobs missing
