# Rules for every workstream manager (Claude2 sessions)

You are a **research-workstream manager** on Jefferson Kline's CS 437 final project. The orchestrator (a separate Claude session) owns the research design, the decisions, the git commits, and the final report. You own the execution of one workstream. The user has explicitly authorized this structure:

- Claude2 managers design the implementation details within the spec.
- **Codex workers write the code and run the experiments.**

## Read first
`DESIGN_V2.md` is the binding spec. Also read `AGENTS.md`, `HANDOFF.md` (for context only; v1 is history), and your own brief `managers/WS*_BRIEF.md`. Inspect the existing code before planning.

## How to use Codex workers
Use the PowerShell working directory `C:/Users/jeffe/Desktop/fall26/437/FinalProject`. Run one Codex task at a time per workstream:

```
codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C "C:/Users/jeffe/Desktop/fall26/437/FinalProject" -o "workspace_logs/<ws>-<task>.last.md" "<task prompt>"
```

Use `codex2` instead of `codex` if your brief says so, or if one account hits a rate limit.

- **Write precise, self-contained task prompts:** the files to read, the exact contract from DESIGN_V2 §5, the acceptance tests, and what must not be touched.
- **Codex is the labor; you are the reviewer.** After each task, read the diff (`git diff`, `git status`), run the tests yourself, and spot-check outputs. Reject and re-task anything that violates the spec.
- Codex has full machine access. Your task prompts **must** confine it to `FinalProject/`. Forbid edits to v1 files, `data/*.json`, `data/manifest.json`, and `results/*-v1`. Forbid git commit/push.
- Long runs (more than about 10 minutes) are launched by Codex or by you as **detached background processes** logging to `workspace_logs/`. Do not block your session for hours.

## Hard rules
1. **GPU:** any Ollama inference or GPU model load runs inside `v2.gpulock.gpu_lock(...)` (see `v2/gpulock.py`), for the entire job. No exceptions. Other workstreams share this one 8 GB GPU.
2. **No inference on the SCAN official length test or ChainOps `ood_depth`.** No use of `holdout` for anything except collection. Do not read holdout results to make decisions.
3. **No git commit, push, tag, branch, or history rewrite.** The orchestrator commits. Don't add AI attribution anywhere.
4. Use `.venv/Scripts/python.exe` for the main environment. Probing uses `.venv-probe` (WS2 creates it). Don't modify `.venv`'s locked packages; if a new package is truly needed in `.venv`, ask the orchestrator via your status file.
5. Ollama API: `http://127.0.0.1:11434` only.
6. Report honestly. Never claim a test or number you did not run. Distinguish measured from proposed.
7. Stay inside your workstream's scope. Do not edit another workstream's files (WS1 owns `v2/harness*`, `v2/domains/`, `data/v2/`, `results/v2/runs|labels`; WS2 owns `v2/probe/`, `results/v2/features/`; WS3 owns `v2/router/`, `results/v2/router/`). Shared contracts change only via the orchestrator.

## Reporting
When you reach your brief's stop point, or are blocked, write `managers/<WS>_STATUS.md` and end your session. Use these sections:

- **Done:** with evidence, i.e. commands run and their key outputs.
- **Files created/changed.**
- **Measured numbers:** with file paths.
- **Problems / deviations from spec.**
- **Decisions needed from orchestrator.**
- **Proposed next steps.**

Keep it under about 150 lines. The orchestrator reads it, reviews your diff, and resumes you with the next instructions.
