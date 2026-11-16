"""Independent artifact readback, including server LFS hashes and dimensions."""
import json
from pathlib import Path
from v2.probe.match_models import api, DEST, ROOT, save, prompt_check

r = json.loads(DEST.read_text())
for m in r['models']:
    mi = m['model_info']
    for c in m['candidates']:
        config = c['documents']['config.json']
        c['parameter_comparison'] = dict(ollama=mi['general.parameter_count'],
            hf_stored=sum(c['parameter_count'].values()),
            equal=mi['general.parameter_count']==sum(c['parameter_count'].values()),
            tie_word_embeddings=config.get('tie_word_embeddings'))
        c['special_token_comparison'] = {key:dict(ollama=mi.get('tokenizer.ggml.'+key), hf=config.get(key)) for key in ['bos_token_id','eos_token_id']}
        c['head_dim_comparison'] = dict(ollama=mi.get('qwen3.attention.key_length'), hf=config.get('head_dim'))
    chosen = next(c for c in m['candidates'] if c['repo_id']==m['chosen_repo_id'])
    assert all(v['equal'] for v in chosen['architecture_comparison'].values())
    assert chosen['parameter_comparison']['equal']
    assert chosen['hybrid_template'] == (not m['chosen_repo_id'].endswith('2507'))
    files = m['download']['files']
    assert sum(f['bytes'] for f in files)==m['download']['total_bytes']
    remote = {s.rfilename:s for s in api.model_info(m['chosen_repo_id'], revision=m['revision'], files_metadata=True).siblings}
    for f in files:
        p = ROOT / 'hf_models' / m['chosen_repo_id'].split('/')[-1] / f['name']
        assert p.stat().st_size == f['bytes']
        if 'sha256' in f:
            assert remote[f['name']].lfs.sha256 == f['sha256'], f['name']
            f['hf_lfs_sha256_verified'] = True
    print(m['ollama_model'], m['chosen_repo_id'], m['revision'], m['download']['total_bytes'], 'PASS')
r['prompt_format_check'] = prompt_check(ROOT / 'hf_models/Qwen3-8B')
assert 'manual_no_think_empty_think_system_blank_line' in r['prompt_format_check']['exact_variants']
r['prompt_format_check']['verified_exact_format'] = '<|im_start|>system\n\n{RULES}<|im_end|>\n<|im_start|>user\n{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
r['independent_readback'] = 'PASS: dimensions, stored parameter counts, hybrid/thinking-only templates, file sizes, and downloaded weight SHA256 against pinned HF LFS metadata.'
save(r)
for name in ['WS2-model-matching.txt', 'WS2-env-check.txt']:
    p = ROOT / 'workspace_logs' / name
    raw = p.read_bytes()
    text = raw.decode('utf-16') if raw.startswith(b'\xff\xfe') else raw.decode('utf-8')
    p.write_text(text.encode('ascii', errors='backslashreplace').decode('ascii'), encoding='ascii')
print(json.dumps(r['prompt_format_check'], indent=2, ensure_ascii=True))
