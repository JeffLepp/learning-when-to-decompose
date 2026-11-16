# Activation probe pipeline

Run from the project root with `.venv-probe/Scripts/python.exe`.

```
python -m v2.probe.extract --domain scan --model qwen3:4b --items data/v2/scan/router_train.jsonl --batch-size 4
python -m v2.probe.v1_features
python -m v2.probe.feasibility_v1
python -m v2.probe.readback
```

The first command imports the harness's `prompt_parts` and its versioned
`<domain>_v0.json` S0 prompt. The harness uses `/api/generate`, `system`,
`prompt`, and `think:false`. Changes to that interface or prompt file selection
must be reflected in the adapter and its equality test.

`extract_features` is CPU-testable and returns float32 pooled arrays in original
row order. Layer k is exactly `hidden_states[k]`, including the final normalized
output as provided by HF. `write_job` accepts explicit items and a custom
message builder and writes float16 arrays atomically. GPU callers must already
hold `gpu_lock` from before model loading until model disposal. The CLI handles
that lock and unloads any resident Ollama models immediately after acquisition.
OOD requests require both `--final-eval` and the existing `freeze-v2` tag.

The v1 runner makes four one-token, nondataset Ollama formatting checks only
when no saved evidence exists. Subsequent execution reuses that evidence and
preserves completed feature sets. It never repeats completed extraction. Run it
via `python -m v2.probe.run_logged v1_features WS2-v1-features.log` to append
ASCII output to the workstream log. Feasibility analysis is CPU-only and uses
only train plus validation for selection. Audit is evaluated after selection.

NF4 loading uses an explicit device map. With the installed versions, automatic
CPU placement after bitsandbytes conversion left NF4 quantization state on the
meta device and failed during the first forward pass. For 8B, the embedding
and output tables are declared CPU before conversion and transformer blocks
are NF4 on GPU. The 4B NF4 comparison fits on GPU. BF16 uses `device_map='auto'`
with a dedicated VRAM budget capped at 5500 MiB and at least 1000 MiB free
headroom. Inputs follow Accelerate's execution device when the stored embedding
weight is on meta. No packages are modified by these scripts.

CPU tests:

```
.venv-probe/Scripts/python.exe -m unittest discover -s v2/tests -t . -p "test_probe_*.py"
.venv/Scripts/python.exe -m v2.probe.safe_discovery
```

The originally requested literal full-suite discovery command ran before an
existing WS1 parser test was found to read the official length-test file. This
violated this worker's no-read rule; no model inference or feature selection
used that file. Its original test summary is preserved in WS2-tests.txt. Do not
repeat that route in this workstream. The scoped safe_discovery runner skips a
test when it tries to open that exact forbidden path, without changing WS1's
test source. Guarded summaries are saved separately in WS2-tests-guarded.txt.
