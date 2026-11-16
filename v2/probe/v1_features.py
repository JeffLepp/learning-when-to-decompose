"""Reproduce v1 prompt IDs, extract the model ladder, and check forced tokens."""
import gc
import json
import time
from pathlib import Path

from v2.gpulock import gpu_lock
from v2.probe.extract import ROOT, api, format_prompt, input_device, load_model, model_info, unload_ollama, write_job

RUNS = [('train-8b-v1','v1_train',96), ('validation-8b-v1','v1_validation',32),
        ('audit-8b-v1','v1_audit',32), ('pilot-8b-v1','v1_dev',24)]
DEST = ROOT / 'results/v2/features/scan_v1'


def records(run):
    return [json.loads(x) for x in (ROOT/'results'/run/'records.jsonl').read_text().splitlines() if x.strip()]


def v1_items():
    items = []
    for run, split, n in RUNS:
        direct = [r for r in records(run) if r['strategy']=='direct']
        assert len(direct)==n
        items.extend(dict(id=r['id'], domain='scan_v1', split=split, input=r['command']) for r in direct)
    assert len(items)==184 and len({i['id'] for i in items})==184
    assert len({i['input'] for i in items})==184
    return items


def messages(item):
    from experiment import RULES
    return RULES, 'Command: ' + item['input'] + '\nReturn {"actions":[...]}.'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True)+'\n', encoding='ascii')


def check_wrappers():
    from transformers import AutoTokenizer
    evidence = {}
    for name in ('qwen3:0.6b','qwen3:1.7b','qwen3:4b','qwen3:8b'):
        info = model_info(name)
        # Read back the live template; preserve it as the formatting evidence.
        shown = api('show', {'model':name})
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'hf_models'/info['chosen_repo_id'].split('/')[-1], local_files_only=True)
        payload = dict(model=name, system='You are terse.', prompt='Say ok.', think=False,
                       stream=False, keep_alive=0, options=dict(temperature=0, seed=437, num_predict=1))
        response = api('generate', payload)
        expected = response['context'][:response['prompt_eval_count']]
        actual = tokenizer(format_prompt(name,payload['system'],payload['prompt']), add_special_tokens=False)['input_ids']
        evidence[name] = dict(match=actual==expected, hf_ids=actual, ollama_ids=expected,
            request=payload, response=response, live_template=shown['template'],
            wall_cost_seconds=response.get('total_duration',0)/1e9)
        save(ROOT/'v2/probe/prompt_format_check.json', evidence)
        print('wrapper',name,actual==expected,len(actual),len(expected),flush=True)
        if actual != expected:
            raise AssertionError('Prompt wrapper mismatch: '+name)


def check_v1(tokenizer, name):
    runs = [r for run,_,_ in RUNS for r in records(run)] if name=='qwen3:8b' else records('pilot-v1')
    rows = [r for r in runs if r['strategy']=='direct']
    matches = 0
    for r in rows:
        text = format_prompt(name,*messages(dict(input=r['command'])))
        ids = tokenizer(text,add_special_tokens=False)['input_ids']
        c = r['calls'][0]
        assert ids==c['context'][:c['prompt_eval_count']], r['id']
        matches += 1
    print('v1 prompt ids',name,matches,'/',len(rows),flush=True)
    return dict(matches=matches,n=len(rows))


def agreement(model, tokenizer, name):
    import torch
    rows = [r for r in records('train-8b-v1' if name=='qwen3:8b' else 'pilot-v1') if r['strategy']=='direct']
    counts = dict(positions=0,matches=0,letter_positions=0,letter_matches=0)
    details = []
    with torch.inference_mode():
        for row in rows:
            c = row['calls'][0]
            ids = c['context']
            n = c['prompt_eval_count']
            inputs = torch.tensor([ids],device=input_device(model))
            output = model(input_ids=inputs,use_cache=False)
            predicted = output.logits[0,n-1:len(ids)-1].argmax(-1).cpu().tolist()
            targets = ids[n:]
            local = dict(id=row['id'],positions=len(targets),matches=0,letter_positions=0,letter_matches=0,
                         recorded_eval_count=c.get('eval_count'))
            for pred, target in zip(predicted, targets):
                ok = int(pred==target)
                local['matches'] += ok
                if any(ch.isalpha() for ch in tokenizer.decode([target])):
                    local['letter_positions'] += 1
                    local['letter_matches'] += ok
            for key in counts:
                counts[key] += local[key]
            details.append(local)
            del output, inputs
    return dict(**counts,agreement=counts['matches']/counts['positions'],
        letter_agreement=counts['letter_matches']/counts['letter_positions'], n_items=len(rows),
        records=details, caveat='Ollama used JSON-schema grammar; HF argmax is unconstrained. Only generated tokens present in recorded context are scored.')


def main():
    import numpy as np
    import torch
    items = v1_items()
    output = json.loads((DEST/'throughput.json').read_text()) if (DEST/'throughput.json').exists() else {}
    agreement_results = json.loads((DEST/'teacher_forced_agreement.json').read_text()) if (DEST/'teacher_forced_agreement.json').exists() else {}
    with gpu_lock('ws2-v1-features', poll=0.5):
        unload_ollama()
        evidence_path = ROOT/'v2/probe/prompt_format_check.json'
        if evidence_path.exists():
            evidence = json.loads(evidence_path.read_text())
            assert len(evidence)==4 and all(e['match'] for e in evidence.values())
            print('Reusing four verified wrapper calls; no repeat Ollama generation.',flush=True)
        else:
            check_wrappers()
        unload_ollama()
        for name, override in [('qwen3:0.6b',None),('qwen3:1.7b',None),('qwen3:4b',None),('qwen3:8b',None),('qwen3:4b','nf4')]:
            folder = name.replace(':','-') + ('-nf4' if override else '')
            dest = DEST/folder
            if dest.exists():
                assert folder in output and json.loads((dest/'index.json').read_text())['ids']==[i['id'] for i in items]
                if name in ('qwen3:1.7b','qwen3:8b'):
                    assert name in agreement_results
                print('Preserving completed',folder,flush=True)
                continue
            start = time.perf_counter()
            model, tokenizer, info, quant = load_model(name,override)
            load_seconds = time.perf_counter()-start
            checks = check_v1(tokenizer,name) if name in ('qwen3:8b','qwen3:1.7b') else None
            index = write_job(model,tokenizer,items,messages,dest,name,info,quant,batch_size=2,
                load_seconds=load_seconds,items_files=[ROOT/'results'/run/'records.jsonl' for run,_,_ in RUNS])
            index['feature_dir_bytes'] = sum(p.stat().st_size for p in dest.iterdir())
            index['v1_prompt_id_check'] = checks
            output[folder] = index
            save(DEST/'throughput.json',output)
            print('THROUGHPUT',folder,json.dumps({k:index[k] for k in ['load_seconds','wall_seconds','items_per_second','tokens_per_second','cuda_max_memory_allocated','cpu_layers','feature_dir_bytes']}),flush=True)
            if name in ('qwen3:8b','qwen3:1.7b'):
                agreement_results[name] = agreement(model,tokenizer,name)
                save(DEST/'teacher_forced_agreement.json',agreement_results)
                print('AGREEMENT',name,agreement_results[name]['agreement'],agreement_results[name]['letter_agreement'],flush=True)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        similarities = {}
        for layer in output['qwen3-4b']['layers']:
            a = np.load(DEST/'qwen3-4b'/f'L{layer:02d}_last.npy').astype(np.float32)
            b = np.load(DEST/'qwen3-4b-nf4'/f'L{layer:02d}_last.npy').astype(np.float32)
            similarities[str(layer)] = float(np.mean((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))))
        save(DEST/'4b_comparison.json',dict(mean_per_item_cosine_last=similarities,
            bf16={k:output['qwen3-4b'][k] for k in ['items_per_second','cuda_max_memory_allocated','cpu_layers']},
            nf4={k:output['qwen3-4b-nf4'][k] for k in ['items_per_second','cuda_max_memory_allocated','cpu_layers']}))


if __name__=='__main__':
    main()
