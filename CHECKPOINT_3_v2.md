# Final Project Milestone 3

**Learning When to Decompose Complex Tasks for Local Language Models**

**Team member:** Jefferson Kline · **Course:** CS 437, Machine Learning

**Code and data:** [github.com/JeffLepp/learning-when-to-decompose](https://github.com/JeffLepp/learning-when-to-decompose)

## 1. Working baseline and initial model

The project asks two questions:

- When does breaking a task into subtasks (decomposition) let a *small, locally run* language model solve tasks it otherwise fails?
- Can a supervised model learn to predict, before any generation, which model-and-strategy combination will succeed most cheaply?

**Pipeline.** The pipeline runs end to end on two tasks:

- **SCAN**, a public benchmark that maps commands such as `jump around left twice and walk` to action sequences;
- **ChainOps**, a synthetic task we generated: chains of list operations such as "Start with [3, 8, 1, 5]. Reverse the list. Then remove every odd number." Because it is generated, no model could have memorized it.

Every item is attempted by four local Qwen3 models (0.6B, 1.7B, 4B, 8B) using each of these strategies:

- **direct** (one answer);
- **few-shot direct**;
- **thinking** (private reasoning before answering);
- **coarse decomposition:** the model plans and solves clause-sized pieces, and code repeats and concatenates them;
- **fine decomposition:** the model parses the full structure and solves only atomic pieces;
- two **diagnostic** conditions that separate planning errors from execution errors. *Gold plan* supplies the correct plan, and the model executes the pieces. *Code execution* lets the model plan, and code executes the pieces.

Diagnostics are never treated as usable strategies. Every call logs its tokens, time, and failures.

**Baselines:**

- The direct and few-shot strategies are the solving baselines.
- The initial learned controller is a logistic-regression and random-forest router. It was trained on a first 8B pilot (96 training, 32 validation, 32 audit commands). Validation selected logistic regression with threshold 0.5. On the audit set it used 6.9% fewer tokens than always decomposing, but answered one fewer command correctly (15/32 vs. 16/32).
- The full router library is built and tested. It reproduces those pilot numbers exactly. It also adds three new input families: word-frequency (TF-IDF) features, activation probes that read the model's hidden states before it generates, and confidence-based cascades.

## 2. Quantitative result and visualization

![Development accuracy by model size and strategy](figures/checkpoint3_dev_accuracy.png)

*Figure regenerated from the label files by `figures/make_checkpoint_fig.py`.*

Development split, exact-match accuracy with 95% Wilson intervals. There are 112 items per cell; thinking was run on 20 items. Tokens are the mean input plus output tokens across all calls.

| Task | Model | Direct | Few-shot | Coarse decomp. | Fine decomp. | Thinking (n=20) | Gold plan (diag.) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SCAN | 0.6B | 0.0% | 1.8% | 0.9% | 1.8% | 0% | 9.8% |
| SCAN | 1.7B | 2.7% | 8.9% | 26.8% | 20.5% | 10% | 34.8% |
| SCAN | 8B | 8.9% [4.9, 15.7] | 36.6% | **97.3% [92.4, 99.1]** | 95.5% | 55% | 100% |
| ChainOps | 0.6B | 7.1% | 1.8% | 9.8% | 8.0% | 40% | 15.2% |
| ChainOps | 1.7B | 17.0% | 13.4% | 18.8% | 44.6% | **95% [76.4, 99.1]** | 48.2% |
| ChainOps | 8B | 33.0% | 38.4% | 41.1% | 91.1% [84.3, 95.1] | 100% | 92.0% |

**Cost.**

- Decomposition at 8B costs about 1,170–1,800 tokens per item, against about 260–370 for direct.
- Thinking costs about 750–1,700 tokens. It is also far slower: about 25 s per SCAN item at 8B, against about 1 s for decomposition.

**ML feasibility result.** The task is predicting whether decomposition will succeed for the 8B model. Data: the 128 pilot training and validation commands, with nested cross-validation and 95% bootstrap intervals.

| Features | AUROC |
| --- | --- |
| Activation probe on the model's hidden state, before generation | 0.855 [0.815, 0.941] |
| TF-IDF word features | 0.834 [0.786, 0.929] |
| Structural counts | 0.787 [0.716, 0.884] |

The probe's layer was chosen on the same data, so its number is optimistic.

## 3. What was run and what we observed

**What was run:**

- The data was split into development, router-training, validation, holdout, and out-of-distribution partitions before any model was run.
- Prompts were tuned on the development split only, within a fixed and logged budget of rounds. Direct and decomposition prompts received comparable effort.
- All model × strategy combinations were then run on the development split.

**What we observed:**

1. **Decomposition transforms capable models.** The 8B model goes from 9% (direct) to 97% (coarse decomposition) on SCAN, and from 33% to 91% (fine decomposition) on ChainOps.
2. **There is a capability floor.** With the correct plan supplied, the 0.6B model still scores only 10% on SCAN, so it cannot reliably execute steps as small as "jump left". At 8B, gold-plan execution is perfect. All of its remaining errors are therefore planning errors.
3. **Thinking competes with decomposition, depending on the task.** On ChainOps, 1.7B with thinking (95%, n=20) matches 8B with decomposition. Model size and inference compute can partly substitute for each other, but not below the floor.
4. **The router's room is mostly in cost, not accuracy.** On development data, a perfect router over the direct and decomposition strategies would add under one point of accuracy over the best single strategy (SCAN 98.2% vs. 97.3%). It would also use 25% fewer tokens (1,056 vs. 1,399). On ChainOps the saving is 36% (1,144 vs. 1,790 tokens). The learned router must capture part of that saving by predicting which items a cheaper model or strategy can handle.
5. **Reading the model's hidden state predicts its failures** better than structural features. This is consistent with recent work (Lugoloobi et al., ICLR 2026). At this sample size, though, it is not clearly better than word-frequency features.

## 4. Important issues discovered

- **A model-release mismatch.** Ollama's `qwen3:4b` turned out to be a later, thinking-only release (2507), not the same Qwen3 release as the other sizes. That would have confounded the size comparison. It was replaced with the original Qwen3-4B; its development results are being rerun and are omitted above.
- **Optimistic numbers.** Development results guided prompt design, so they are optimistic. Headline claims will come only from a fresh 400-item holdout and from out-of-distribution tests (longer SCAN commands, deeper ChainOps chains), each evaluated once after the design is frozen.
- **Weak direct baselines on SCAN.** Direct prompting stayed weak on SCAN despite three logged prompt-improvement rounds. SCAN is known to be hard to solve in one pass, but the gap should be read as "these prompts", not as a limit of the models.
- **Plan validity.** Small models often produce malformed plans: 32–48% of coarse ChainOps plans at 0.6B and 1.7B are invalid. These count as failures. They are a real weakness of small models, partly amplified by strict format checking.
- **Thinking runs out of budget.** Thinking often hits its 2,048-token cap on SCAN (40% of 8B attempts). Truncation is reported as part of that strategy's cost.
- **Cost measurement.** Token counts are a cost proxy, not energy or FLOPs. Wall-clock times on a single shared GPU are reported but treated as secondary.
- **Routers often fail.** Many published routers do not beat simply using the best single strategy (LLMRouterBench, 2026). Our router is judged against that baseline and against a perfect-router upper bound, so a null result is possible and would be reported as one.

## 5. Next step

1. Rerun the 4B model on the development split and freeze all prompts.
2. Collect the main label set: about 1,200 items per task across all model × strategy combinations, roughly 17 GPU-hours.
3. Extract activation features for every model size.
4. Train and select routers on the training and validation splits only. Compare structural, TF-IDF, and activation-probe features, plus confidence-based cascades, against the best single strategy and the perfect-router bound. Report learning curves and calibration.
5. Freeze the design. Then evaluate once on the holdout and on the out-of-distribution splits.
6. Write the final analysis of decomposition and thinking as two ways of trading model size for inference compute.

## References

- Lake and Baroni (2018), [Generalization without systematicity](https://arxiv.org/abs/1711.00350); [SCAN dataset](https://github.com/brendenlake/SCAN).
- Zhou et al. (2022), [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625). Decomposition on SCAN is prior work; this project studies small models and learned strategy selection.
- Lugoloobi et al. (ICLR 2026), [LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](https://arxiv.org/abs/2602.09924).
- [LLMRouterBench](https://arxiv.org/abs/2601.07206) (2026).
- Yue et al. (ICLR 2024), [LLM Cascades with Mixture of Thought Representations](https://arxiv.org/abs/2310.03094).
- Prasad et al. (NAACL Findings 2024), [ADaPT: As-Needed Decomposition and Planning](https://arxiv.org/abs/2311.05772).
- Snell et al. (2024), [Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314).
