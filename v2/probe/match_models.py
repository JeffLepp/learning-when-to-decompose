"""Anonymous, CPU-only checkpoint matching, download and prompt readback."""
import os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, get_safetensors_metadata

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'v2/probe/model_matching.json'
api = HfApi(token=False)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(obj):
    DEST.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + '\n', encoding='ascii')


def ollama(route, payload=None):
    req = urllib.request.Request('http://127.0.0.1:11434/api/' + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def candidate(repo):
    info = api.model_info(repo, files_metadata=True)
    docs = {}
    for name in ['config.json', 'tokenizer_config.json', 'generation_config.json']:
        p = hf_hub_download(repo, name, revision=info.sha, token=False)
        docs[name] = json.loads(Path(p).read_text(encoding='utf-8'))
    metadata = get_safetensors_metadata(repo, revision=info.sha, token=False)
    return dict(repo_id=repo, revision=info.sha, documents=docs,
                parameter_count=metadata.parameter_count,
                files=[dict(name=s.rfilename, size=s.size) for s in info.siblings])


def prompt_check(directory):
    from transformers import AutoTokenizer
    from experiment import RULES
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    rows = []
    for line in (ROOT / 'results/train-8b-v1/records.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['strategy'] == 'direct':
            rows.append(r)
        if len(rows) == 20:
            break
    counts = {}
    differences = []
    for row in rows:
        c = row['calls'][0]
        target = c['context'][:c['prompt_eval_count']]
        texts = {}
        texts['hf_enable_thinking_false'] = tokenizer.apply_chat_template(
            [{'role':'system','content':RULES}, {'role':'user','content':c['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for toggle in ['', ' /no_think']:
            for suffix in ['', '<think>\n\n</think>\n\n']:
                key = 'manual' + ('_no_think' if toggle else '_no_toggle') + ('_empty_think' if suffix else '_no_block')
                texts[key] = '<|im_start|>system\n' + RULES + '<|im_end|>\n<|im_start|>user\n' + c['prompt'] + toggle + '<|im_end|>\n<|im_start|>assistant\n' + suffix
                texts[key + '_system_blank_line'] = texts[key].replace('<|im_start|>system\n', '<|im_start|>system\n\n', 1)
        for key, text in texts.items():
            ids = tokenizer.encode(text, add_special_tokens=False)
            counts[key] = counts.get(key, 0) + int(ids == target)
            if key == 'hf_enable_thinking_false' and ids != target and not differences:
                n = next((i for i,(a,b) in enumerate(zip(ids,target)) if a != b), min(len(ids),len(target)))
                differences.append(dict(first_difference=n, hf_length=len(ids), ollama_length=len(target),
                    hf_decoded_tail=tokenizer.decode(ids[max(0,n-10):]),
                    ollama_decoded_tail=tokenizer.decode(target[max(0,n-10):])))
    return dict(n_records=len(rows), exact_match_counts=counts, differences=differences,
        construction='<|im_start|>system\n{RULES}<|im_end|>\n<|im_start|>user\n{prompt}{toggle}<|im_end|>\n<|im_start|>assistant\n{suffix}',
        variants=dict(toggle=['', ' /no_think'], suffix=['', '<think>\n\n</think>\n\n'], system_newlines=[1,2]),
        exact_variants=[k for k,v in counts.items() if v == len(rows)])


def main():
    result = dict(models=[], anonymous=True, inference_performed=False)
    tags = {m['name']:m for m in ollama('tags')['models']}
    fields = {'qwen3.block_count':'num_hidden_layers', 'qwen3.embedding_length':'hidden_size',
        'qwen3.feed_forward_length':'intermediate_size', 'qwen3.attention.head_count':'num_attention_heads',
        'qwen3.attention.head_count_kv':'num_key_value_heads', 'qwen3.context_length':'max_position_embeddings'}
    for size in ['0.6', '1.7', '4', '8']:
        name = 'qwen3:' + size + 'b'
        show = ollama('show', {'model':name, 'verbose':True})
        mi = {}
        for k,v in show['model_info'].items():
            mi[k] = dict(length=len(v), sha256=hashlib.sha256(json.dumps(v, ensure_ascii=True, separators=(',',':')).encode()).hexdigest(), hash_encoding='compact ensure_ascii JSON UTF-8') if isinstance(v,list) else v
        tensors = show.get('tensors', [])
        entry = dict(ollama_model=name, digest=tags[name]['digest'], details=show['details'],
            model_info=mi, template=show['template'], parameters=show.get('parameters'),
            tensor_summary=dict(count=len(tensors), examples=tensors[:6]), candidates=[])
        result['models'].append(entry)
        repos = sorted({m.id for m in api.list_models(author='Qwen', search='Qwen3-' + size + 'B')
                        if m.id in {'Qwen/Qwen3-' + size + 'B', 'Qwen/Qwen3-' + size + 'B-Instruct-2507', 'Qwen/Qwen3-' + size + 'B-Thinking-2507'}})
        entry['official_variant_search'] = repos
        for repo in repos:
            cand = candidate(repo)
            config = cand['documents']['config.json']
            cand['architecture_comparison'] = {k:dict(ollama=mi.get(k), hf=config.get(v), equal=mi.get(k)==config.get(v)) for k,v in fields.items()}
            cand['vocab_comparison'] = dict(ollama=mi.get('tokenizer.ggml.tokens',{}).get('length'), hf=config.get('vocab_size'))
            cand['parameter_comparison'] = dict(ollama=mi.get('general.parameter_count'),
                hf_stored=sum(cand['parameter_count'].values()),
                equal=mi.get('general.parameter_count')==sum(cand['parameter_count'].values()),
                tie_word_embeddings=config.get('tie_word_embeddings'))
            cand['special_token_comparison'] = {key:dict(ollama=mi.get('tokenizer.ggml.'+key), hf=config.get(key)) for key in ['bos_token_id','eos_token_id']}
            cand['head_dim_comparison'] = dict(ollama=mi.get('qwen3.attention.key_length'), hf=config.get('head_dim'))
            cand['hybrid_template'] = 'enable_thinking' in cand['documents']['tokenizer_config.json'].get('chat_template','')
            entry['candidates'].append(cand)
        chosen_id = 'Qwen/Qwen3-' + size + 'B'
        if mi.get('general.version') == '2507' and mi.get('general.finetune') == 'Thinking':
            chosen_id += '-Thinking-2507'
        chosen = next(c for c in entry['candidates'] if c['repo_id'] == chosen_id)
        entry['chosen_repo_id'], entry['revision'] = chosen['repo_id'], chosen['revision']
        entry['evidence'] = ('Explicit Thinking/2507 metadata and official license URL, 262144 context and thinking-only template identify Thinking-2507; original hybrid has 40960 context.' if chosen_id.endswith('2507') else 'Original hybrid checkpoint: architecture, parameter count, vocabulary, context and thinking-toggle template compared to official alternatives.')
        entry['uncertainty'] = 'Ollama GGUF Q4_K_M cannot be bitwise compared to original BF16 safetensors; matching architecture and metadata do not prove weight lineage. Probe NF4/BF16 differs from execution Q4_K_M.'
        if chosen_id.endswith('2507'):
            entry['uncertainty'] += ' This tag is a thinking-only July release, unlike the original hybrid model ladder assumed by the design; orchestrator review is required before extraction or claims about a homogeneous ladder.'
        save(result)
        directory = ROOT / 'hf_models' / chosen['repo_id'].split('/')[-1]
        print('Downloading', chosen['repo_id'], chosen['revision'], flush=True)
        snapshot_download(chosen['repo_id'], revision=chosen['revision'], token=False, local_dir=directory,
            allow_patterns=['*.safetensors', '*.safetensors.index.json', 'config.json', 'generation_config.json', 'tokenizer*', 'vocab.json', 'merges.txt', 'added_tokens.json', 'special_tokens_map.json', 'chat_template*'], max_workers=4)
        files = [p for p in directory.rglob('*') if p.is_file() and '.cache' not in p.parts]
        entry['download'] = dict(total_bytes=sum(p.stat().st_size for p in files), files=[dict(name=p.relative_to(directory).as_posix(), bytes=p.stat().st_size, **({'sha256':sha(p)} if p.suffix=='.safetensors' else {})) for p in sorted(files)])
        save(result)
        print('Verified bytes', entry['download']['total_bytes'], flush=True)
    result['prompt_format_check'] = prompt_check(ROOT / 'hf_models/Qwen3-8B')
    save(result)
    print(json.dumps(result['prompt_format_check'], indent=2), flush=True)


if __name__ == '__main__':
    main()
