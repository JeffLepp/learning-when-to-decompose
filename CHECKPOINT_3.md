# Final Project Milestone 3

**Learning When to Decompose Complex Tasks for Local Language Models**  
**Team member:** Jefferson Kline  
**Course:** CS 437 — Machine Learning

## Working baseline and initial model

This project investigates whether a supervised controller can learn when a local language model benefits from breaking a task into subtasks. The objective is to balance exact-match accuracy with execution cost.

A working pipeline uses the public SCAN dataset, which maps compositional commands to action sequences. The direct baseline produces the complete sequence in one model response. The decomposition baseline asks the same model to create an ordered plan, solves each clause in a separate call, and uses code to repeat and concatenate the resulting actions. Code checks output format but does not repair semantic errors using the reference answer.

Initial development experiments used Qwen3 1.7B and Qwen3 8B. The controller experiment used local Qwen3 8B with Q4_K_M quantization and thinking disabled. Logistic regression and random forests were fitted to predict each strategy's success using command length and word-count features. A separate regressor predicted token costs. There were 96 controller-training commands, 32 validation commands, and 32 separate audit commands, all drawn from disjoint partitions of the official training split. Validation selected logistic regression with a probability threshold of 0.5. The official test set has not been used for model inference.

## Quantitative result

The following results are from the 32-command audit:

| Strategy | Correct commands | Exact-match accuracy | Mean total tokens per command |
| --- | ---: | ---: | ---: |
| Direct-only | 3/32 | 9.4% | 257.3 |
| Decomposition-only | 16/32 | 50.0% | 671.4 |
| Learned controller | 15/32 | 46.9% | 625.1 |

Total tokens include input and output tokens across all calls in the chosen strategy. The controller used **6.9% fewer tokens** than always decomposing but answered one fewer command correctly. It chose decomposition for 28 commands and direct execution for four. These results show an initial accuracy/cost tradeoff, not a clear improvement over both baselines.

## Observations and important issues

Decomposition substantially improved accuracy over the current direct prompt, but both execution strategies still made semantic errors. For example, the decomposed plan for `look twice after walk thrice` executed the clauses in the wrong order, although direct execution answered that development example correctly. Other failures occurred when an individual subtask placed an action before a required turn. Valid JSON therefore did not guarantee a correct answer.

The preset 80% validation-accuracy target was not reached. The small audit has considerable uncertainty, and the direct baseline needs stronger prompting before drawing broad conclusions about decomposition. Token counts are a cost proxy rather than a direct measurement of computational work. Recorded controller latency replays selected executions and does not establish live end-to-end routing speed. Six pipeline/controller tests passed, and fresh file readback reproduced scores, token totals, and all saved audit decisions.

## Next step

Strengthen both strategies using development data, including a fair few-shot direct baseline. Compare direct solving, model-generated plans, and correct supplied plans with the 1.7B and 8B models to distinguish planning failures from subtask-solving failures. Supplied plans will be a diagnostic condition, not a deployable competitor. Then collect more labels, compare learned routing with a simple heuristic, and evaluate on a fresh holdout before using the official test set. This preserves the supervised-learning objective while testing how decomposition affects small-model capabilities.

## References

- Lake and Baroni (2018), [Generalization without systematicity](https://arxiv.org/abs/1711.00350); [official SCAN dataset](https://github.com/brendenlake/SCAN).
- Zhou et al., [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625). Decomposition on SCAN is established prior work; this project studies learned strategy selection for local models.
