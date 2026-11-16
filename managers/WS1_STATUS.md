# WS1 status: phase A (P0, P1, P2) complete; stopped before P3

Written 2026-11-15 by the WS1 manager. Nothing is committed. No inference touched `holdout`, `ood_length`, `ood_depth`, `router_train` or `router_val`; all inference was on `dev`.

## Done (with evidence)

- **P0 foundations** (Codex task, report `workspace_logs/ws1-p0.last.md`; I read all of `harness.py`, `scan.py`, `chainops.py` and reran the tests).
  - `.venv/Scripts/python.exe -m unittest discover -s v2/tests -t .` → `Ran 77 tests ... OK (skipped=3)` (includes WS2/WS3 tests). WS1-only: `-m unittest v2.tests.test_foundations v2.tests.test_promptdev v2.tests.test_thinking_prefill` → `Ran 17 tests ... OK`.
  - The SCAN interpreter and the independent expand+code-leaf path both reproduce every official train and test line. ChainOps templates round-trip. Perfect-mock runs score 100% on all 7 arms × 2 domains. OOD/holdout refusal, resume idempotency and config-change refusal are tested.
  - `data/v2/` built: 128 / 600 / 200 / 400 / 300 per domain (3,256 items), hashes in `data/v2/manifest.json`, OOD files marked locked.
- **P1 prompt development** (Codex task, report `workspace_logs/ws1-p1.last.md`). I recomputed all 20 log lines from the raw records: they match. Scored on 60 fixed dev items with `qwen3:1.7b` and `qwen3:8b`; 16-id exemplar pool excluded from all scoring.
- **P2 timing pilot + final-prompt dev scoring** (run by me through the harness, under the GPU lock): `dev-final-<domain>` (112 dev items × 4 models × S0,S1,S3,S4,G1,G2) and `pilot-s2b-<domain>` (20 dev items × 4 models × S2).
- **Label tables** for those dev runs: `python -m v2.build_labels --allow-runs dev-final-scan,dev-final-chainops,pilot-s2b-scan,pilot-s2b-chainops` → 2,768 rows per domain.

## Files created/changed

- Code: `v2/__init__.py`, `v2/harness.py`, `v2/build_data.py`, `v2/build_labels.py`, `v2/harness_promptdev.py`, `v2/domains/{__init__,scan,chainops}.py`, `v2/tests/{test_foundations,test_promptdev,test_thinking_prefill}.py`.
- Prompts: `v2/domains/prompts/{scan,chainops}_{v0,v0p,v1,final}.json`, `scan_v2.json`.
- Data: `data/v2/**`.
- Results: `results/v2/runs/{smoke-*-p0, pd-*, dev-final-*, pilot-s2-*, pilot-s2b-*}`, `results/v2/prompt_dev/**`, `results/v2/labels/{scan,chainops}.csv`.
- Scratch (gitignored): `workspace_logs/ws1-*`, including `ws1_p2_tables.py` and its output `ws1-p2-tables.txt`.

## Final prompts

| File | SHA-256 | Blocks selected (round) |
| --- | --- | --- |
| `v2/domains/prompts/scan_final.json` | `d587e10e7af83de388e274014d460cc75faedb245e109322137deb9ee13cb5d8` | S0←0, S1←1, S3←1, S4←0, S2=S0 |
| `v2/domains/prompts/chainops_final.json` | `2eaaa69b1d8d110bdf2f4f2c054614d8ec30e379c37b152e14d1d252efdc8b34` | all round 0, S2=S0 |

Iteration log: `results/v2/prompt_dev/log.jsonl` (prompt and block hashes, per-model dev scores, rationales); selection in `selection.json`. SCAN used 3 of 5 rounds, ChainOps 2 of 5; each stopped at the first round that improved no strategy.

## Measured numbers

Source: `results/v2/runs/dev-final-*` and `pilot-s2b-*`; table script output in `workspace_logs/ws1-p2-tables.txt`. Dev items only; these are not generalization estimates. S2 has n=20, all other arms n=112. The 60 prompt-selection items are a subset of the 112.

**Dev accuracy, SCAN (%)**

| Model | S0 | S1 | S2 (n=20) | S3 | S4 | G1 | G2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6b | 0.0 | 1.8 | 0.0 | 0.9 | 1.8 | 9.8 | 27.7 |
| 1.7b | 2.7 | 8.9 | 10.0 | 26.8 | 20.5 | 34.8 | 56.2 |
| 4b | 8.9 | 16.1 | 5.0 | 44.6 | 82.1 | 100.0 | 82.1 |
| 8b | 8.9 | 36.6 | 55.0 | 97.3 | 95.5 | 100.0 | 95.5 |

**Dev accuracy, ChainOps (%)**

| Model | S0 | S1 | S2 (n=20) | S3 | S4 | G1 | G2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6b | 7.1 | 1.8 | 40.0 | 9.8 | 8.0 | 15.2 | 48.2 |
| 1.7b | 17.0 | 13.4 | 95.0 | 18.8 | 44.6 | 48.2 | 83.0 |
| 4b | 18.8 | 8.0 | 85.0 | 30.4 | 72.3 | 74.1 | 92.0 |
| 8b | 33.0 | 38.4 | 100.0 | 41.1 | 91.1 | 92.0 | 99.1 |

**Timing: mean / median seconds per item, mean total tokens**

| Model | Arm | SCAN s | SCAN tok | ChainOps s | ChainOps tok |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6b | S0 | 0.46 / 0.29 | 241 | 0.22 / 0.22 | 372 |
| 0.6b | S1 | 1.70 / 0.44 | 1260 | 0.24 / 0.20 | 747 |
| 0.6b | S2 | 2.51 / 2.07 | 651 | 3.71 / 3.24 | 1010 |
| 0.6b | S3 | 0.64 / 0.55 | 1042 | 0.53 / 0.49 | 1296 |
| 0.6b | S4 | 1.16 / 1.02 | 1151 | 0.75 / 0.68 | 1782 |
| 1.7b | S0 | 0.36 / 0.33 | 239 | 0.21 / 0.21 | 374 |
| 1.7b | S1 | 1.23 / 0.47 | 1258 | 0.19 / 0.18 | 737 |
| 1.7b | S2 | 10.06 / 11.02 | 1649 | 3.98 / 3.62 | 925 |
| 1.7b | S3 | 0.62 / 0.63 | 1363 | 0.42 / 0.41 | 1112 |
| 1.7b | S4 | 0.93 / 0.93 | 1148 | 0.78 / 0.76 | 1807 |
| 4b | S0 | 1.01 / 0.72 | 273 | 0.60 / 0.43 | 380 |
| 4b | S1 | 0.89 / 0.87 | 1162 | 0.39 / 0.36 | 730 |
| 4b | S2 | 22.84 / 23.27 | 2208 | 12.79 / 10.68 | 1471 |
| 4b | S3 | 1.37 / 1.40 | 1322 | 1.14 / 0.98 | 1340 |
| 4b | S4 | 1.92 / 1.95 | 1181 | 1.90 / 1.82 | 1813 |
| 8b | S0 | 1.04 / 0.61 | 263 | 0.36 / 0.32 | 369 |
| 8b | S1 | 1.18 / 0.75 | 1170 | 0.38 / 0.38 | 735 |
| 8b | S2 | 24.52 / 24.46 | 1702 | 6.65 / 5.25 | 753 |
| 8b | S3 | 1.10 / 1.06 | 1399 | 1.10 / 1.03 | 1439 |
| 8b | S4 | 1.46 / 1.45 | 1173 | 1.50 / 1.44 | 1790 |

G1 costs 0.36–1.08 s and G2 adds no GPU time when S4 runs in the same run (it reuses the parse call). Full G1/G2 rows are in the tables file.

**Failure rates**

- **S2 truncation at the 2048 cap:** SCAN 0% / 30% / 95% / 40% (0.6b / 1.7b / 4b / 8b); ChainOps 5% / 0% / 15% / 0%. On SCAN, 4B S2 is almost entirely cap-limited (mean 2,010 thinking tokens).
- **Non-thinking truncation at 512:** SCAN S1 16.1% (0.6b), 15.2% (1.7b), 2.7% (8b); SCAN S0 0.9–4.5%; ChainOps 0%.
- **Format failures:** 0% in every cell.
- **Invalid plans:** ChainOps S3 32.1% / 48.2% / 25.0% / 2.7%; SCAN S3 35.7% (0.6b), 2.7% (1.7b); ChainOps S4 10.7% (0.6b); otherwise ≤1.8%.

**Budget, extrapolated from the per-arm GPU seconds above**

| Scope | GPU-hours |
| --- | ---: |
| P3 as planned in §7, both domains | 19.7 (SCAN 12.7, ChainOps 7.0) |
| of which S2 | 9.7 (SCAN 6.7, ChainOps 3.0) |
| P5 OOD (300 main + 100 S2 per domain) | 4.7, a lower bound: OOD items are longer |
| Total | about 24.4, above the ~20 target |

Disk: 25 KB per record measured (`dev-final-scan` is 67 MB for 2,688 records, with every call duplicated in `calls.jsonl`). P3's roughly 51,000 records would be about 1.3 GB or more.

## Problems / deviations from spec

1. **Harness bug found in P2 and fixed by me, not Codex.** `qwen3:4b` emits no opening `<think>` token, so the thinking/content split halted both `pilot-s2-*` runs at the 4B model. Fix is 3 lines in `token_parts` plus `v2/tests/test_thinking_prefill.py`, replayed on the two real failing calls (2048/0 and 888/11 thinking/content tokens). Consequences: `pilot-s2-*` are partial, superseded by `pilot-s2b-*`; `dev-final-*` configs record the pre-fix harness hash (the fix only affects `think:true` calls). Current `v2/harness.py` SHA-256 starts `d4c7ef84`.
2. **P2 shape.** Instead of 20 items × 7 arms in one run, the six non-thinking arms were timed on 112 items and S2 on 20 in a separate run, so S2 was not interleaved with the others.
3. **Direct baselines stay weak** despite the fairness rule: SCAN S0 ≤ 8.9% for every model, S1 36.6% at 8B. Prompt rounds 1–2 did not improve S0.
4. **ChainOps S3 is handicapped by plan validity**, not only by execution: 25–48% invalid plans for 0.6b–4b (chunks failing the "one or two sentences joined by ` Then `" structural check).
5. **`min_logprob` is dominated by grammar-forced JSON tokens** (values near −38 on the first content token after thinking). F4 confidence features should not use raw min without care.
6. **Thinking-token definition:** the thinking phase includes the `<think>`/`</think>` delimiters; one unreported end token per call stays in `completion_tokens` only. `total_tokens` includes thinking, as D1.4 requires.
7. **Minor.** ChainOps text has "Then Keep only…" capitalisation and "the first 1 elements"; left alone because fixing it changes data hashes. One 1-clause SCAN dev item (`walk opposite left`, in the 112 but not the 60) equals a sub-clause shown inside a pool exemplar; such overlap is inherent to SCAN.
8. ChainOps has 14 operations (all listed candidates), not "about 12".

## Decisions needed from orchestrator

1. **Budget.** Recommend halving the S2 subsets (75 train / 50 val / 75 holdout / 50 OOD): saves about 4.8 h in P3 and 1.2 h in P5, giving about 18.4 GPU-h total. Alternative: keep sizes and accept about 24.4 h.
2. **S2 cap on SCAN.** 95% truncation for 4B means S2 there measures the cap. Raising it needs a larger `num_ctx` too (prompt + 4096 exceeds 4096) and roughly doubles S2 cost. Recommend keeping 2048 and reporting truncation as the result.
3. **More prompt rounds?** SCAN has 2 unused rounds and ChainOps 3. The obvious targets are ChainOps S3 plan validity and the direct prompts. Caveat: I have now seen the 112-item results, so further rounds are no longer blind to them (dev only, but worth recording).
4. **Approve the final prompts** above (or the result of item 3) so they can be tagged frozen.
5. **Storage.** Decide whether `results/v2/runs/**` is committed. If so, consider dropping the duplicate `calls.jsonl` and the per-call `context` arrays; that is a harness change and needs new run names.
6. Whether to fix the ChainOps wording nits (item 7) before P3 and before WS2 extracts features from `data/v2`.

## Proposed next steps

1. On approval, launch P3 model-major as detached runs, one per domain × split, in the order `router_train`, `router_val`, then `holdout` with `--collect-holdout`; S2 and G1/G2 subsets via `--ids-file` (stratified by depth).
2. Build `results/v2/labels/*.csv` from P3 runs only; the current CSVs hold dev rows and should be regenerated then.
3. Hand WS3 the dev label tables now as a format check against real data.
