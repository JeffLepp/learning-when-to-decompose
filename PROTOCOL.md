# Exploratory protocol — 2026-11-15

This project implements Jefferson Kline's Milestone 2 proposal, preserved in `references/milestone2-user-provided.txt`; the original PDF was not found. The user subsequently supplied the five checkpoint requirements in `references/checkpoint3-user-requirements.txt`. `CHECKPOINT_3.md` addresses them using these exploratory results. This protocol describes the original pilot, not proposed expansion experiments or final test evaluation.

## Fixed experiment

- Dataset: official SCAN length split, pinned by Git commit and SHA-256 hashes in `data/manifest.json`.
- Entire official training pool partitioned before inference with seed 437: 128 development, 10,000 controller training, 3,000 validation, 3,862 audit commands. Official length test: 3,920 commands, downloaded for split-integrity checks but not evaluated with the model.
- Development pilot: first 24 commands, paired strategies, initially Qwen3 1.7B and then Qwen3 8B. Development results informed model choice only.
- Controller pilot: first 96 controller-training, 32 validation, 32 audit commands from the fixed randomized partitions. Same Qwen3 8B model digest, prompts, decoding, and schemas throughout. No prompt edits after model selection.
- Decoding: temperature 0, seed 437, thinking disabled, context 4,096, maximum 512 generated tokens per call. JSON schema output for both strategies. These choices define this experiment; they do not establish optimal model performance.
- Direct: one call returns the complete action sequence.
- Decomposed: one call creates an ordered plan with one or two clauses and repeat counts; a separate call solves each clause; code repeats and concatenates actions. Direction and around/opposite remain model reasoning tasks. There is no reference-answer access during execution, automatic semantic repair, retry, or oracle fallback.
- Format failures count as incorrect. Transport/server failures stop collection instead of creating mislabeled examples. A semantically wrong but well-formed plan is still possible.
- One warmup per run, excluded from reported metrics; alternating direct-first and decomposition-first ordering. Full requests, responses, completion status, timing, and token counts are retained. Model weights and dataset hashes are recorded. Run configuration changes require a new output directory.

## Controller and selection

The inputs are word/character length and counts of primitive commands, directions, conjunctions, and modifiers. Embeddings are deferred. Two binary classifiers predict direct and decomposed success. Compare standardized logistic regression with a shallow random forest. A constant classifier handles a one-class training label safely.

A separate random forest predicts each strategy's input-plus-output token cost from the same prompt features. Choose the cheapest predicted strategy whose predicted success reaches threshold t. If neither qualifies, choose the strategy with higher predicted success (direct wins an exact tie). This fallback is not a success guarantee.

Threshold grid: 0, .25, .5, .6, .7, .8, .9, .95. Select classifier family and threshold using validation only: lowest realized mean token cost among candidates with at least 80% validation accuracy. If no candidate reaches 80%, report the target as unmet and select highest validation accuracy, then lowest token cost. No audit-driven retuning in this experiment.

The audit reports direct-only, decomposition-only, selected controller, and an oracle accuracy ceiling. The oracle sees recorded correctness and is only an upper bound, never a deployable strategy. The selected model and per-command predictions are saved. An audit used to change future modeling decisions becomes development evidence; final claims require a fresh holdout.

## Measurement limits

The primary cost proxy is total prompt plus generated tokens, with both components also saved in execution summaries. This is not FLOPs, energy, or an assumption that input and output tokens cost equally in practice. Wall time covers the whole selected execution strategy including all its calls and local validation/composition. Ollama server timing and loading time are also retained per call. Prompt caching is enabled by the server and may favor later calls; counterbalancing reduces but does not eliminate this effect.

Controller runtime is a replay of measured strategy runtimes, not a fresh live routed benchmark. Its reported batch decision time excludes feature extraction and cost estimation. Final evaluation should time full live routing, repeat trials, report uncertainty on paired differences, and account for the cost of generating training labels. Wilson accuracy intervals on this tiny audit are descriptive, not proof of superiority.

The first 1.7B pilot used `localhost`, which had about 2.05 seconds of connection delay in a diagnostic request. Later runs use `127.0.0.1`, which took about 0.001 seconds. Do not compare their raw wall times as a model speedup. Original source is archived with each run.

The length test measures length extrapolation. An additional composition split, such as add-primitive jump, is still required for the proposal's unseen-composition claim, with a separately trained controller and disjoint evaluation. Do not mix overlapping official splits into one training pool. SCAN is public and synthetic: pretrained-model exposure is unknown, and success on SCAN does not establish improvement on general complex tasks.

## Sources

- [Official SCAN dataset](https://github.com/brendenlake/SCAN) and its bundled license.
- [Lake and Baroni (2018), Generalization without systematicity](https://arxiv.org/abs/1711.00350).
- [Zhou et al., Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625). Prior work already studies decomposition on SCAN; this project's intended contribution is cost-sensitive learned strategy selection for a local model. Its model and prompting differ from this pilot, so published accuracy is not a directly comparable baseline.
- [Ollama generate API](https://docs.ollama.com/api/generate), for response timing and token fields.
