# Final project session instructions

Read `HANDOFF.md` first, then `CHECKPOINT_3.md`, `RESULTS.md`, and `PROTOCOL.md`. Inspect relevant code before proposing changes. Treat the measured JSON/JSONL records as the source of truth for numerical claims.

The user confirmed: prepare this checkpoint from verified results FIRST, then expand the decomposition experiments. No deadline planning is needed and no submission format or length was specified. The five checkpoint requirements are saved verbatim in `references/checkpoint3-user-requirements.txt`. Do not ask for these again.

Preserve the machine-learning objective: a supervised controller learns when to use direct versus decomposed local-model execution. The user is especially interested in what small models can accomplish with decomposition and wants rigorous work that supports a strong grade. Do not promise a grade or positive results. Distinguish measured findings from proposed experiments.

Keep `HANDOFF.md` current after meaningful decisions, implementation, and results, including unfinished obligations. Make your own plan from the preserved state; suggested experiments are not evidence that they have already been run. Do not rerun completed collection merely to reconstruct context.

Use `.venv/Scripts/python.exe` within this directory. The parent course `.venv` is broken. Use the Ollama HTTP API at `http://127.0.0.1:11434`; the earlier `localhost` route added connection delay, and the installed `ollama.exe` launched the app rather than behaving as the expected list CLI. Recheck server/model availability before future inference. The readback and unit tests do not need a running model.

Preserve existing run folders, raw records, data partitions, and archived executor code. Use new names for new run configurations. Keep the official test set out of prompt/controller development. Once audit results influence redesign, use a fresh holdout for subsequent claims. Record all model calls and their costs, failures, and code assistance; a correct supplied plan is a diagnostic condition, not a deployable strategy.

**v2 expansion (authorized by the user 2026-11-15):** `DESIGN_V2.md` is the binding spec. The user explicitly authorized a multi-agent structure. A Claude orchestrator owns design, decisions, git commits, and the report. Claude2 workstream managers (`managers/*_BRIEF.md`, `managers/COMMON.md`) direct Codex/Codex2 workers, who implement and run the experiments. The user also approved model and PyTorch downloads, unattended full-auto agents confined to `FinalProject/`, a local-only git repo (no remote, commits only by the orchestrator under the user's identity, no AI attribution), and a second domain (ChainOps). All GPU work must hold `v2/gpulock.py`.

Finish authorized work through relevant validation, delivery, and independent readback. Keep explanations concise and ask only for consequential missing information. Spawn agents only within the authorized v2 workstream structure above. Do not send messages, submit, publish, commit, or move the project to another drive as part of merely preparing the checkpoint or handoff. If future Git work is authorized, use the existing human author/committer identity and no assistant attribution trailers.
