# Learning When to Decompose Complex Tasks for Local Language Models

Jefferson Kline — exploratory work toward Final Project Milestone 3, November 15, 2026.

**Fresh session:** read [HANDOFF.md](HANDOFF.md) and [AGENTS.md](AGENTS.md) first. The user confirmed checkpoint preparation first, followed by expanded experiments; no deadline planning or particular submission format/length is required.

The checkpoint is prepared in [CHECKPOINT_3.md](CHECKPOINT_3.md), covering all five [user-supplied requirements](references/checkpoint3-user-requirements.txt). The original Milestone 2 PDF was not found; its supplied text is preserved in [references/milestone2-user-provided.txt](references/milestone2-user-provided.txt). The CS 415 Milestone 2 on the E: drive is a separate project.

## What is implemented

- Official SCAN length-split data, pinned to an upstream commit with file hashes and disjoint development/training/validation/audit partitions.
- Local Ollama execution with Qwen3 1.7B and Qwen3 8B development pilots; the controller experiment uses Qwen3 8B Q4_K_M.
- Direct solving versus model-generated decomposition, individual subtask solving, and deterministic repetition/concatenation.
- Strict intermediate format validation, exact-match scoring, full response traces, token counts, wall time, model identity, and frozen run configurations.
- Logistic-regression and random-forest success predictors using prompt length and command-structure features, plus a learned token-cost estimator.
- Validation-only model/threshold selection and a separate exploratory audit. Embedding features are not implemented in this first experiment.

See [RESULTS.md](RESULTS.md) for measured results and interpretation, and [PROTOCOL.md](PROTOCOL.md) for partition sizes, selection policy, sources, and limitations. The official 3,920-command length test has not been evaluated by the model.

## Files

| File | Purpose |
| --- | --- |
| `experiment.py` | Download/check data and collect paired execution labels |
| `controller.py` | Train both classifier families, choose threshold, evaluate audit, save selected model |
| `test_experiment.py`, `test_controller.py` | Six tests for parsing, validation, combination, features, and routing |
| `verify_results.py` | Independent file readback, score/token checks, and saved-model prediction reproduction |
| `results/*/records.jsonl` | Per-command raw model responses and measurements |
| `results/*/config.json`, `experiment.py` | Run configuration and archived executor source |
| `results/controller-v1/report.json` | Full validation search and audit metrics |
| `results/controller-v1/controller.joblib` | Selected fitted controller and cost estimator |
| `data/manifest.json` | Source commit, hashes, partition sizes |

## Run it

Ollama must be running at `127.0.0.1:11434` with `qwen3:8b` installed. From this directory in PowerShell:

```powershell
.\reproduce.ps1
.\.venv\Scripts\python.exe verify_results.py
```

The setup script uses `uv` and the project-local Python 3.14 environment with pinned dependencies. It checks failures at each step. Existing matching run files are resumed/reused; it does not silently replace collected data. To collect a fresh repetition, supply new `--name` values to the commands below and pass those names to the controller:

```powershell
.\.venv\Scripts\python.exe experiment.py run --model qwen3:8b --partition controller_train --count 96 --name train-repeat
.\.venv\Scripts\python.exe experiment.py run --model qwen3:8b --partition validation --count 32 --name validation-repeat
.\.venv\Scripts\python.exe experiment.py run --model qwen3:8b --partition audit --count 32 --name audit-repeat
.\.venv\Scripts\python.exe controller.py --train train-repeat --validation validation-repeat --audit audit-repeat --name controller-repeat
```

The first 1.7B pilot is historical development evidence; its archived executor used the slower `localhost` connection. Do not compare its raw runtime against later runs as a model speed comparison. Python's removed 3.13 interpreter made the older course-wide `.venv` unusable, so this project has its own verified environment and setup script. Existing coursework was not edited.

## Work remaining toward the checkpoint and final project

1. The current checkpoint text is prepared from verified results in `CHECKPOINT_3.md`; it has not been submitted. Any future submission or format conversion is a separate user request.
2. Improve execution prompts using development data only. Include a fair few-shot direct baseline and investigate planning-order errors, turn/action-order errors, and truncation. Do not treat the present weak direct baseline as competitive prompting.
3. Generate more training labels and measure learning curves. Compare learned routing with a simple fixed heuristic; assess calibration and uncertainty. Freeze the revised procedure before opening a new holdout.
4. Evaluate the official length test once the design is frozen. Add an independently partitioned composition-generalization experiment, with no overlapping commands leaking across training and test.
5. Measure live routed latency, repeated-run variability, total label-generation cost, and full controller overhead. Current controller latency figures replay recorded executions.

Decomposition on SCAN is established prior work. The research question here is whether learned local-model routing achieves a useful accuracy/cost tradeoff. A small positive pilot or a negative result is preliminary evidence, not proof of general agent improvement.
