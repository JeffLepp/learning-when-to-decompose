# Task E completed

Implemented and validated the synthetic fixture generator, P4 development driver, frozen P5 driver,
cross-fitted calibration, batched bootstrap, independent readback, and acceptance tests.
Task D work was preserved. No GPU, Ollama, network, installs, Git mutation, or real v2 final-item reads.

## Verified results

| Data / split | Selected router | Accuracy | Tokens | Normalized AUC |
|---|---|---:|---:|---:|
| v1 validation | route/F1/logreg/per_arm | 0.3125 | 601.625 | 0.228632 |
| Synthetic validation | route/F1/logreg/per_arm | 0.9750 | 342.905 | 0.930092 |
| Synthetic holdout | route/F1/logreg/per_arm | 0.9400 | 361.405 | 0.915493 |
| Synthetic OOD length | route/F1/logreg/per_arm | 0.9800 | 853.333 | 0.627126 |

Validation was used for selection and is optimistic. Synthetic results are fixtures, not capability evidence.
The v1 best single reached 0.3125 accuracy at 688.03125 tokens; selected routing used 601.625 tokens.

## Validation and timing

- 57 tests passed, zero skipped, Python exit code 0; baseline was 50 passing tests.
- Historical `v1_repro`: `match: true` (15/32 audit, 625.09375 tokens).
- Independent readback passed for both P4 runs and all synthetic final/S2 reports.
- Source/input/frozen hashes and identical recompressed payloads verified; both final repeats refused.
- Both Mode-A domains reran with the correction; only their summaries remain.
- Full command wall times: v1 P4 40.83s; synthetic P4 233.79s; synthetic final 19.37s.
- Additional statistics-only correction: 33.88s; no execution-model refits or policy reselection.

## Corrections and provenance

Independent readback exposed floating-point endpoint ties incorrectly producing null AUCs.
The batched metric now handles numerical ties, with a dedicated regression test and end-to-end readback test.
All 76,098 selected decision rows and 6,447 aggregate curve points stayed unchanged.
Corrected 3 missing observed AUC values; genuine unreachable-target CIs remain undefined.
Original reports and affected sources are archived in `pre_metric_fix/`. `fit_code_sha256` preserves
fit-time provenance; `code_sha256` records the corrected frozen evaluation code. The one-shot evaluation
logs still contain one completed call per split. No final evaluation call was repeated to repair statistics.
Validation cross-fit calibrators were recreated from unchanged raw predictor outputs to replay curves;
final policies used the saved full-validation calibrators without fitting.
Frozen joblib files were losslessly transcoded to LZMA; original serialized payload hashes match exactly.
Original report archives and container hashes preserve provenance. JSON whitespace was compacted losslessly.
Summary table escaping was corrected; crowded synthetic figures were rerendered from saved metrics.

## Artifacts and limits

- [v1 summary](../p4-v1adapter/summary.md), [synthetic summary](../synthetic16/p4/summary.md).
- [Synthetic holdout](../synthetic16/p4/final_holdout/summary.md), [synthetic OOD](../synthetic16/p4/final_ood_length/summary.md).
- [Mode-A SCAN](../mode_a_scan/summary.md), [Mode-A ChainOps](../mode_a_chainops/summary.md).
- `acceptance.json`: numbers, CIs, wall times, hash/readback checks and final size.
- `tests.log`, `statistics_correction.json`, `packaging.json`, `json_compaction.json`: audit evidence.
- [Git status](git_status.txt) includes pre-existing workstream changes; no commits or other Git writes were made.
- Token cost is a replay proxy. HF/Ollama quantization differs. Missing adapter S0 costs use WS2 probe-token counts.
- Bootstrap conditions on fitted models/choices; readback independently checks decisions/costs and aggregate AUC, not classifier fitting or every bootstrap CI.

No unfinished Task E obligations remain.
