# WS2 status: phase A complete (stop point reached)

Date: 2026-11-15. Worker: `codex2`, two tasks (T1 environment/matching/downloads, T2 pipeline/tests/runs). Prompts and worker reports: `workspace_logs/WS2-T1.*`, `workspace_logs/WS2-T2.*`. Everything below marked "manager-verified" I re-ran or read back myself after the worker finished; the rest is the worker's measurement, read from its saved files.

## Done

1. **`.venv-probe/`**: Python 3.14.10 (official cp314 cu128 wheels exist, so no second interpreter was needed). torch 2.11.0+cu128, CUDA 12.8, `sm_120` in arch list, transformers 5.23.0, accelerate 1.15.0, bitsandbytes 0.50.2, numpy 2.5.5, scikit-learn 1.9.1. GPU matmul and a 4-bit nf4 linear forward pass passed under `gpu_lock` (`workspace_logs/WS2-env-check.txt`). Lock file: `requirements-probe-lock.txt`.
2. **Model matching**: table below; evidence in `v2/probe/model_matching.json`. Manager-verified against `/api/show`.
3. **`v2/probe/extract.py`**: forward-only extraction, per-model Ollama prompt formatting, OOD guard, atomic output directory, CLI and Python entry point. `v2/probe/prompts.py` imports WS1's `v2.harness.prompt_parts` (no prompt text copied).
4. **Feasibility probe on v1 labels**: `v2/probe/feasibility_v1.py` -> `results/v2/features/scan_v1/feasibility_report.json`, table in `workspace_logs/WS2-feasibility.md`.
5. **Tests**: `v2/tests/test_probe_extract.py`. Manager-verified: `.venv-probe` -> `Ran 6 tests ... OK`; `.venv` (no torch) -> `Ran 6 tests ... OK (skipped=3)`.

## Model matching

| Ollama | HF checkpoint @ revision | Layers / hidden | Params | Forward dtype |
| --- | --- | --- | ---: | --- |
| qwen3:0.6b | Qwen/Qwen3-0.6B @ c1899de2 | 28 / 1024 | 751,632,384 | bf16 |
| qwen3:1.7b | Qwen/Qwen3-1.7B @ 70d244cc | 28 / 2048 | 2,031,739,904 | bf16 |
| qwen3:4b | **Qwen/Qwen3-4B-Thinking-2507** @ 768f209d | 36 / 2560 | 4,022,468,096 | bf16 (see problem 2) |
| qwen3:8b | Qwen/Qwen3-8B @ b968826d | 36 / 4096 | 8,190,735,360 | 4-bit nf4 |

- Architecture, parameter count, vocabulary and context length match for all four. Downloaded safetensors sha256 match HF LFS metadata. Total 30.06 GB in `hf_models/`. Ollama weights are Q4_K_M, so no bitwise comparison is possible.
- **Mismatch in the ladder (manager-verified):** `qwen3:4b` reports `finetune=Thinking`, `version=2507`, context 262144; the other three are the April 2025 hybrid release (context 40960). Its Ollama template has no `/no_think` toggle and always ends the prompt with `<|im_start|>assistant\n<think>\n`, even with `think:false`.
- **Prompt format, exact token-id match with Ollama:**
  - 0.6b / 1.7b / 8b: `<|im_start|>system\n\n{system}<|im_end|>\n<|im_start|>user\n{user} /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. HF's own `enable_thinking=False` template matched 0/20; this string matched 184/184 v1 8B direct prompts and 24/24 v1 1.7B prompts.
  - 4b: `<|im_start|>system\n{system}\n\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n`.
  - All four confirmed by one 1-token Ollama call each on a non-dataset string (`v2/probe/prompt_format_check.json`, all `match: true`, manager-verified).
- **Teacher-forced agreement** (HF argmax vs. the token Ollama generated in v1, forward passes only; Ollama was grammar-constrained, HF was not): 8B 4907/5025 = 97.65%; 1.7B 563/623 = 90.37% (`results/v2/features/scan_v1/teacher_forced_agreement.json`).

## Measured numbers

Extraction on the 184 v1 SCAN commands (v1 direct prompt, about 185 tokens each), batch size 2, warm-up excluded. Source: each `results/v2/features/scan_v1/<model>/index.json` (manager-verified: shapes `[184, H]`, float16, all finite, 184 unique ids).

| Model | Forward dtype | Load s | Forward s | Items/s | Tokens/s | Peak VRAM (torch) | Dir size |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6b | bf16 | 10.8 | 7.8 | 23.56 | 4371 | 1.25 GiB | 5.1 MB |
| 1.7b | bf16 | 12.4 | 8.0 | 22.91 | 4251 | 3.37 GiB | 10.1 MB |
| 4b | bf16, 13 of 36 layers on CPU | 26.1 | 85.3 | 2.16 | 387 | 5.23 GiB | 16.2 MB |
| 4b | nf4 (comparison) | 28.3 | 15.9 | 11.56 | 2076 | 2.84 GiB | 16.2 MB |
| 8b | nf4 | 63.2 | 74.7 | 2.46 | 457 | 5.05 GiB | 25.9 MB |

- Layers saved: every 4th plus last (7 layers for 28-layer models, 9 for 36-layer), two poolings each.
- Size scales linearly: for 1,500 items per domain the 8B directory would be about 211 MB.
- 4b bf16 vs. nf4, mean per-item cosine of last-token features: 0.996 at L04 falling to 0.926 at L36 (`4b_comparison.json`).

Feasibility (v1 8B labels, selection set = v1 train + validation, n=128, 50 decomposed-success positives, 9 direct-success positives). Nested CV, 5 folds x 5 repeats, C chosen in an inner 3-fold; 95% item-bootstrap CIs.

| Target | Feature | CV AUROC | 95% CI |
| --- | --- | ---: | --- |
| decomposed | probe L32 mean (best) | 0.855 | 0.815-0.941 |
| decomposed | probe L28 last (best last-token) | 0.805 | 0.727-0.885 |
| decomposed | F2 TF-IDF | 0.834 | 0.786-0.929 |
| decomposed | F1 structural (v1 features) | 0.787 | 0.716-0.884 |
| direct (9 positives) | probe L28 last (best) | 0.838 | 0.688-0.965 |
| direct | F2 TF-IDF | 0.795 | 0.440-0.929 |
| direct | F1 structural | 0.631 | 0.401-0.848 |

- Paired difference, best probe minus F1: +0.076 (CI 0.019 to 0.140). Best probe minus F2: +0.020 (CI -0.028 to 0.070). Both are optimistically biased because the best layer was picked on the same 128 items.
- **Exploratory only (v1 audit was already inspected; n=32):** decomposed-success AUROC probe L32 mean 0.902 (0.777-0.984), F1 0.856, F2 0.824. Direct-success: probe 0.713 (0.097-1.0), F1 0.828, F2 0.839; with so few positives these intervals say nothing.
- Manager cross-check with an independent simple 5-fold fit (fixed C=0.01): L32 mean 0.855, L08 mean 0.862, L36 last 0.777, consistent with the table.
- **Reading:** the probe is feasible and clearly beats structural features, but it is not distinguishable from TF-IDF at this sample size. Mean pooling beat last-token for the decomposed target.

## Files created/changed

- `requirements-probe-lock.txt`; `.venv-probe/` and `hf_models/` (both gitignored).
- `v2/probe/`: `extract.py`, `prompts.py`, `v1_features.py`, `feasibility_v1.py`, `match_models.py`, `verify_matching.py`, `check_env.py`, `readback.py`, `run_logged.py`, `test_logged.py`, `safe_discovery.py`, `model_matching.json`, `prompt_format_check.json`.
- `v2/tests/test_probe_extract.py`.
- `results/v2/features/scan_v1/`: `qwen3-0.6b/`, `qwen3-1.7b/`, `qwen3-4b/`, `qwen3-4b-nf4/`, `qwen3-8b/`, plus `throughput.json`, `4b_comparison.json`, `teacher_forced_agreement.json`, `feasibility_report.json`, `execution_failures.json`, `call_costs.json`, `readback.json`.
- No WS1/WS3 files, v1 files, or `.venv` touched. No commits.

## Problems / deviations from spec

1. **`qwen3:4b` is a different release (thinking-only 2507).** The size ladder is not one model family at one release, and S0 with `think:false` on 4b still starts inside an open `<think>` block. This affects WS1's arms as well as F3.
2. **4B does not fit in 8 GB in bf16.** As specified (bf16) it runs with 13 layers on CPU at 2.16 items/s; nf4 runs fully on GPU at 11.56 items/s, but its late-layer features differ (cosine 0.93 at L32-L36).
3. **Feature file layout** is one 2-D file per layer and pooling, `L{layer:02d}_{pooling}.npy`, listed under `files` in `index.json`, instead of a single `<layer>.npy`. `index.json` uses the key `ids` (D1.3). WS3's loader (`v2/router/data.py`) already reads `files`, `ids`, `n_tokens` and the same `qwen3-8b` directory naming (colon is illegal in Windows paths).
4. **`v2/probe/prompts.py` is pinned to `v2/domains/prompts/<domain>_v0.json`.** WS1 now also has `*_final.json`. It must point at whichever file is frozen before any v2 extraction.
5. The 8B nf4 load first failed (bitsandbytes with automatic offload); fixed with an explicit device map (embeddings and output head on CPU, all blocks on GPU). Because of the restart, the five extractions ran across two lock-holding processes instead of one. No partial feature directory was written.
6. The worker ran the full `v2/tests` suite twice, which executes WS1's interpreter test that reads the official length-test file. No model saw that file and nothing was selected with it. WS1's brief allows that read for the interpreter check, so I treat this as minor; the worker added a guard for its later runs.
7. `hf_models/Qwen3-4B` (5 GB, partial original-release download) is unused. Left in place.
8. No v2-domain features were extracted: v2 items and frozen prompts were not final when this ran. Throughput was measured on v1 prompts only; batch size 2 was not tuned.

## Decisions needed from orchestrator

1. **4B rung:** keep Ollama `qwen3:4b` (Thinking-2507) and report the heterogeneity, or replace it with a GGUF of the original hybrid Qwen3-4B so the ladder is one release. I recommend replacing it if WS1 has not started P3 collection, since it changes what S0 and S2 mean for that rung.
2. **4B probe dtype:** bf16 with CPU offload (to spec, about 5x slower: roughly 23 min per 3,000 items) or nf4 (consistent with 8B, fast). I recommend bf16: the run is short and it avoids a second quantization caveat.
3. **Which prompt file is frozen** for S0 (`*_final.json`?), so `prompts.py` can be pointed at it.
4. **Confirm the feature layout** in problem 3 as the contract.
5. **Holdout features:** may WS2 extract `holdout` activations before the freeze (forward pass only, no labels read), or only after?

## Proposed next steps

1. After decisions 1-3: point `prompts.py` at the frozen prompts, enable the skipped adapter test, and extract `dev`, `router_train`, `router_val` for both domains and all four models. Estimated GPU time from the table above: under 1 hour total for 800 items x 2 domains, dominated by 4B and 8B.
2. Re-run the per-layer probe on real v2 labels with layer choice on `router_val` only.
3. Try a larger batch size for 0.6b/1.7b only if extraction time becomes a constraint (it currently is not).
