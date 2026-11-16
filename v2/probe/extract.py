"""Forward-only activation extraction with ordered, atomic artifacts."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HYBRID = '<|im_start|>system\n\n{system}<|im_end|>\n<|im_start|>user\n{user} /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
THINKING = '<|im_start|>system\n{system}\n\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n'


def wrapper(model):
    if model not in ('qwen3:0.6b', 'qwen3:1.7b', 'qwen3:4b', 'qwen3:8b'):
        raise ValueError('Unknown Ollama model: ' + model)
    return THINKING if model == 'qwen3:4b' else HYBRID


def format_prompt(ollama_model, system, user):
    text = wrapper(ollama_model).format(system=system, user=user)
    if not system:
        text = text[text.index('<|im_start|>user'):]
    return text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api(endpoint, payload=None):
    request = urllib.request.Request('http://127.0.0.1:11434/api/' + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def unload_ollama():
    for model in api('ps').get('models', []):
        api('generate', {'model': model['name'], 'keep_alive': 0})


def guard_items(items, final_eval=False):
    if any(i['split'] in ('ood_length', 'ood_depth') for i in items):
        frozen = subprocess.run(['git', 'tag', '-l', 'freeze-v2'], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        if not final_eval or frozen != 'freeze-v2':
            raise ValueError('OOD items require --final-eval and git tag freeze-v2')


def input_device(model):
    embedding = model.get_input_embeddings()
    hook = getattr(embedding, '_hf_hook', None)
    if hook is not None and hasattr(hook, 'execution_device'):
        return hook.execution_device
    return embedding.weight.device


def extract_features(model, tokenizer, texts, layers, batch_size):
    import numpy as np
    import torch
    if batch_size < 1 or not texts:
        raise ValueError('Require nonempty texts and positive batch_size')
    ids = [tokenizer(t, add_special_tokens=False)['input_ids'] for t in texts]
    if any(not x for x in ids):
        raise ValueError('Empty token sequence')
    lengths = np.array([len(x) for x in ids], dtype=np.int64)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    last = mean = None
    device = input_device(model)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            width = max(len(ids[i]) for i in indices)
            tokens = torch.full((len(indices), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(tokens)
            for row, i in enumerate(indices):
                offset = width - len(ids[i]) if tokenizer.padding_side == 'left' else 0
                tokens[row, offset:offset + len(ids[i])] = torch.tensor(ids[i], device=device)
                mask[row, offset:offset + len(ids[i])] = 1
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            output = model(input_ids=tokens, attention_mask=mask, position_ids=positions,
                           output_hidden_states=True, use_cache=False)
            if last is None:
                last = np.empty((len(layers), len(ids), output.hidden_states[0].shape[-1]), dtype=np.float32)
                mean = np.empty_like(last)
            for j, layer in enumerate(layers):
                h = output.hidden_states[layer].float()
                real = mask.to(h.device)
                end = (real * torch.arange(width, device=h.device)).max(-1).values
                last[j, indices] = h[torch.arange(len(indices), device=h.device), end].cpu().numpy()
                mean[j, indices] = ((h * real.unsqueeze(-1)).sum(1) / real.sum(1).unsqueeze(-1)).cpu().numpy()
            del output
    return last, mean, lengths


def model_info(name):
    return next(m for m in json.loads((ROOT / 'v2/probe/model_matching.json').read_text())['models']
                if m['ollama_model'] == name)


def load_model(name, quant=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    info = model_info(name)
    path = ROOT / 'hf_models' / info['chosen_repo_id'].split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    quant = quant or ('nf4' if name == 'qwen3:8b' else 'bf16')
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16, device_map='auto',
                  attn_implementation='sdpa')
    # Reserve activation and desktop headroom; never use shared GPU memory.
    free, total = torch.cuda.mem_get_info()
    cap = min(5500 * 1024**2, free - 1000 * 1024**2)
    if cap < 1024**3:
        raise RuntimeError('Insufficient free dedicated VRAM')
    kwargs['max_memory'] = {0: int(cap), 'cpu': '24GiB'}
    if quant == 'nf4':
        # An explicit map is required: transformers converts linear modules before
        # resolving an 'auto' map, leaving offloaded NF4 quant_state on meta.
        # Declare the CPU tables before conversion; transformer blocks stay NF4
        # on GPU, and the two unquantized tables are offloaded by Accelerate.
        kwargs['device_map'] = ({'model.embed_tokens':'cpu', 'model.layers':0,
                                'model.norm':0, 'lm_head':'cpu'}
                               if name == 'qwen3:8b' else {'':0})
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return model, tokenizer, info, quant


def write_job(model, tokenizer, items, message_builder, output_dir, ollama_model,
              info=None, quant='bf16', batch_size=4, layers=None, load_seconds=0,
              items_files=(), overwrite=False, final_eval=False, warmup=True):
    import numpy as np
    import torch
    import transformers
    guard_items(items, final_eval)
    dest = Path(output_dir)
    if dest.exists() and not overwrite:
        raise FileExistsError(str(dest))
    if len({i['id'] for i in items}) != len(items):
        raise ValueError('Duplicate item ids')
    messages = [message_builder(i) for i in items]
    texts = [format_prompt(ollama_model, s, u) for s, u in messages]
    count = model.config.num_hidden_layers
    layers = layers or sorted(set(range(4, count + 1, 4)) | {count})
    if warmup:
        extract_features(model, tokenizer, texts[:batch_size], layers, batch_size)
    cuda = any(p.device.type == 'cuda' for p in model.parameters())
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last, mean, n_tokens = extract_features(model, tokenizer, texts, layers, batch_size)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    info = info or {}
    files = {f'L{layer:02d}_{pool}': f'L{layer:02d}_{pool}.npy' for layer in layers for pool in ('last', 'mean')}
    index = dict(ids=[i['id'] for i in items], splits=[i['split'] for i in items],
        n_tokens=n_tokens.tolist(), domain=items[0]['domain'], ollama_model=ollama_model,
        hf_model_id=info.get('chosen_repo_id', 'random-test-model'), revision=info.get('revision'),
        quantization=quant, dtype='bfloat16' if quant in ('bf16','nf4') else quant,
        saved_dtype='float16', layers=layers, poolings=['last','mean'], files=files,
        hidden_size=last.shape[-1], n_layers=count,
        prompt_template=dict(wrapper=wrapper(ollama_model), messages=[dict(
            system_sha256=hashlib.sha256(s.encode()).hexdigest(), user_sha256=hashlib.sha256(u.encode()).hexdigest()) for s,u in messages]),
        items_files=[dict(path=str(p), sha256=sha(p)) for p in items_files],
        code_sha256=sha(__file__), batch_size=batch_size, wall_seconds=elapsed,
        load_seconds=load_seconds, total_prompt_tokens=int(n_tokens.sum()),
        items_per_second=len(items)/elapsed, tokens_per_second=int(n_tokens.sum())/elapsed,
        cuda_max_memory_allocated=torch.cuda.max_memory_allocated() if cuda else 0,
        device_map={str(k):str(v) for k,v in getattr(model,'hf_device_map',{}).items()},
        cpu_layers=sum(str(v)=='cpu' and 'layers.' in k for k,v in getattr(model,'hf_device_map',{}).items()),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        timestamp=datetime.now(timezone.utc).isoformat(), warmup_excluded=warmup)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=dest.name + '.tmp-', dir=dest.parent))
    try:
        for j, layer in enumerate(layers):
            for pool, values in [('last',last),('mean',mean)]:
                np.save(temp / files[f'L{layer:02d}_{pool}'], values[j].astype(np.float16))
        (temp/'index.json').write_text(json.dumps(index, indent=2, ensure_ascii=True)+'\n', encoding='ascii')
        if dest.exists():
            backup = dest.with_name(dest.name + '.previous')
            if backup.exists():
                raise FileExistsError(str(backup))
            dest.rename(backup)
            try:
                temp.rename(dest)
            except BaseException:
                backup.rename(dest)
                raise
            shutil.rmtree(backup)
        else:
            temp.rename(dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return index


def main():
    from v2.gpulock import gpu_lock
    from v2.probe.prompts import s0_messages
    p = argparse.ArgumentParser()
    p.add_argument('--domain', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--items', nargs='+', required=True, type=Path)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--quant', choices=['nf4'])
    p.add_argument('--final-eval', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    items = [json.loads(line) for path in args.items for line in path.read_text().splitlines() if line.strip()]
    guard_items(items, args.final_eval)
    if any(i['domain'] != args.domain for i in items):
        raise ValueError('Domain mismatch')
    dest = ROOT/'results/v2/features'/args.domain/args.model.replace(':','-')
    if dest.exists() and not args.overwrite:
        raise FileExistsError(str(dest))
    # Validate the adapter before acquiring resources.
    for item in items:
        s0_messages(item)
    with gpu_lock('ws2-extract'):
        unload_ollama()
        t0 = time.perf_counter()
        model, tokenizer, info, quant = load_model(args.model, args.quant)
        write_job(model, tokenizer, items, s0_messages, dest, args.model, info, quant,
                  args.batch_size, load_seconds=time.perf_counter()-t0, items_files=args.items,
                  overwrite=args.overwrite, final_eval=args.final_eval)
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
