# Preliminary experiment results

November 15, 2026. These are exploratory results toward Milestone 3, not final test results. The user subsequently supplied the checkpoint requirements; the prepared response is in [CHECKPOINT_3.md](CHECKPOINT_3.md). The original proposal PDF remains unavailable; its supplied text is preserved.

## Main finding

The pipeline runs end to end, but this pilot does **not** establish that learned routing improves on always decomposing. On 32 held-out audit commands, the controller used 6.9% fewer total tokens than decomposition-only while getting one fewer answer correct. Overall execution accuracy remains too low to meet the preset 80% validation target. Improving the execution prompts is the next priority.

## Audit: 32 previously unused training-pool commands

The local execution model was Qwen3 8B Q4_K_M with thinking disabled, on an NVIDIA GeForce RTX 5060 (8 GB). Each command was run with both strategies. Controller training used 96 different commands; model/threshold selection used 32 different validation commands. The official SCAN length test remains unused for inference.

| Strategy | Exact match | Accuracy | Mean input + output tokens | Mean recorded/replayed seconds |
| --- | ---: | ---: | ---: | ---: |
| Direct-only | 3/32 | 9.4% | 257.3 | 1.086 |
| Decomposition-only | 16/32 | 50.0% | 671.4 | 0.876 |
| Learned controller | 15/32 | 46.9% | 625.1 | 0.837 |

The controller chose decomposition for 28 commands and direct execution for four. Relative to decomposition-only, those switches gained one correct answer and lost two. This is a cost/accuracy tradeoff, not dominance. The mean token reduction was 46.3 tokens per command, or 6.9%.

The accuracy intervals are wide: controller 30.9–63.6%, decomposition-only 33.6–66.4%, and direct-only 3.2–24.2% (marginal 95% Wilson intervals). These are not paired significance tests. The best possible correctness-aware choice between the two recorded outputs would solve 18/32 (56.25%); this oracle ceiling shows that routing alone cannot repair most execution failures in this audit.

Latency requires care. Direct execution's median was 0.497 seconds versus 0.861 for decomposition, but several long direct generations raised its mean; its maximum was 6.729 seconds. Decomposition's maximum was 1.240 seconds. Controller latency is a replay of selected recorded executions, excludes full routing overhead, and is not an independently timed live deployment result. Input-plus-output token totals also are not equivalent to compute or energy.

## Controller selection

Both logistic regression and random forests were fitted using command length and lexical structure counts. Each predicted success separately for direct and decomposed execution. A random-forest regressor estimated token costs. Embeddings were deferred.

Validation selected **logistic regression with threshold 0.5**. It achieved 10/32 (31.25%) validation accuracy at 629.9 mean tokens, compared with 10/32 and 688.0 tokens for always decomposing. No candidate met the 80% validation target; the predeclared fallback selected highest accuracy, then lowest token cost. A predicted threshold of 0.5 is not a promise that the resulting system achieves 50% accuracy or that an individual answer is correct.

All candidate settings, Brier scores, selected settings, audit predictions, and the fitted model are in [results/controller-v1](results/controller-v1). No audit-driven retuning was performed.

## Development evidence and error analysis

On the same 24 development commands:

| Model | Direct correct | Decomposed correct | Invalid decomposition outputs |
| --- | ---: | ---: | ---: |
| Qwen3 1.7B | 1/24 | 0/24 | 17/24 |
| Qwen3 8B | 1/24 | 13/24 | 1/24 |

The first run exposed planning-format problems: repeated commands often retained `twice`/`thrice` while also assigning a repeat count. Validation rejected these plans. The larger model produced more valid and useful plans, motivating its selection for this pilot. These are prompt-specific results, not a model ranking. The first run also had approximately two seconds of `localhost` connection overhead per request; later runs use `127.0.0.1`, so raw runtime comparisons between the two model pilots would be confounded.

Saved 8B development traces illustrate three distinct cases:

- **Planning error:** `look twice after walk thrice` was correct directly. The decomposed plan executed look before walk, reversing the required order. Format validation could not catch this semantic error.
- **Subtask error:** for `jump left and look left`, the plan split correctly, but the look subtask placed the look action before the left turn. Both strategies failed exact match.
- **Useful decomposition:** `run opposite left twice and run` was solved correctly by planning, solving the smaller clauses, and repeating/combining them. Direct execution produced the wrong turn/repetition sequence.

The 96 training commands yielded 8 direct successes and 40 decomposition successes. Both solved six commands; direct alone solved two; decomposition alone solved 34; neither solved 54. Three direct responses were truncated, and three decomposition plans were invalid. The small and imbalanced success-label set limits classifier learning and probability calibration.

## Delivered and verified

- Runnable experiment, classifiers, saved selected model, pinned data, frozen run metadata/source, and raw responses.
- Six passing tests covering SCAN parsing, malformed intermediate outputs, code combination, prompt-only features, and threshold/cost routing.
- Fresh readback verified all **416 strategy records** across five runs: unique pairs, source commands/targets, exact-match labels, token totals, summary scores/times, and archived-source hashes.
- Reloading the delivered controller reproduced all 32 saved audit decisions and their correctness/token aggregates. See [results/verification.json](results/verification.json).
- Dataset and partition hashes and disjointness passed verification. No model inference ran on the official test set.

## Next checkpoint work

The user's five checkpoint requirements are now preserved in `references/checkpoint3-user-requirements.txt` and addressed in `CHECKPOINT_3.md`. They specified checkpoint preparation first, followed by expansion, with no deadline planning or format/length constraint. The checkpoint has not been submitted.

Use development-only commands to strengthen both execution strategies, including a fair few-shot direct baseline and an ablation of thinking mode. Generate a larger label set after freezing the revised executor; compare a simple routing heuristic, logistic regression, and random forests. Use a fresh holdout for revised experiments rather than reusing this audit to claim generalization. Then evaluate official length and composition splits separately, and measure live routed latency and total label-generation cost.

The research contribution should be learned local-model strategy selection; decomposition on SCAN itself is prior work. See [PROTOCOL.md](PROTOCOL.md) for the sources, fixed experimental choices, and remaining measurement gaps.
