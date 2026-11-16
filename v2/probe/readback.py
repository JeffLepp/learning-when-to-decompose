"""Independent readback of completed WS2 feature and analysis artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from v2.probe.v1_features import DEST, ROOT, v1_items


def main():
    items=v1_items()
    throughput=json.loads((DEST/'throughput.json').read_text())
    output={}
    for folder,row in throughput.items():
        path=DEST/folder
        index=json.loads((path/'index.json').read_text())
        assert index['ids']==[i['id'] for i in items]
        assert index['splits']==[i['split'] for i in items]
        assert len(index['n_tokens'])==184
        assert sum(index['n_tokens'])==index['total_prompt_tokens']
        assert abs(index['items_per_second']*index['wall_seconds']-184)<1e-6
        assert abs(index['tokens_per_second']*index['wall_seconds']-index['total_prompt_tokens'])<1e-6
        for source in index['items_files']:
            assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
        checks={}
        for layer in index['layers']:
            for pool in index['poolings']:
                key=f'L{layer:02d}_{pool}'
                array=np.load(path/index['files'][key])
                assert array.shape==(184,index['hidden_size'])
                assert array.dtype==np.float16 and np.isfinite(array).all()
                checks[key]=hashlib.sha256((path/index['files'][key]).read_bytes()).hexdigest()
        output[folder]=dict(n=184,n_files=len(checks),array_sha256=checks,
                            bytes=sum(p.stat().st_size for p in path.iterdir()))
        assert output[folder]['bytes']==row['feature_dir_bytes']
    assert len(output)==5
    prompts=json.loads((ROOT/'v2/probe/prompt_format_check.json').read_text())
    assert len(prompts)==4 and all(x['match'] and x['hf_ids']==x['ollama_ids'] for x in prompts.values())
    agreements=json.loads((DEST/'teacher_forced_agreement.json').read_text())
    for model,a in agreements.items():
        assert a['n_items']==(96 if model=='qwen3:8b' else 24)
        split='v1_train' if model=='qwen3:8b' else 'v1_dev'
        assert {r['id'] for r in a['records']}=={i['id'] for i in items if i['split']==split}
        for key in ['positions','matches','letter_positions','letter_matches']:
            assert sum(r[key] for r in a['records'])==a[key]
        assert a['matches']/a['positions']==a['agreement']
        assert a['letter_matches']/a['letter_positions']==a['letter_agreement']
    comparison=json.loads((DEST/'4b_comparison.json').read_text())
    for layer,expected in comparison['mean_per_item_cosine_last'].items():
        name=f'L{int(layer):02d}_last.npy'
        a=np.load(DEST/'qwen3-4b'/name).astype(np.float32)
        b=np.load(DEST/'qwen3-4b-nf4'/name).astype(np.float32)
        cosine=float(np.mean((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))))
        assert abs(cosine-expected)<1e-7
    report=json.loads((DEST/'feasibility_report.json').read_text())
    for target,block in report['targets'].items():
        for key,row in block['table'].items():
            assert len(row['repeat_averaged_oof_scores'])==128
            assert len(row['chosen_C_per_outer_fold'])==25
            assert abs(np.mean(row['repeat_aurocs'])-row['mean_oof_auroc'])<1e-12
        best=max((k for k in block['table'] if k.startswith('L')),key=lambda k:block['table'][k]['mean_oof_auroc'])
        assert best==block['cv_best_probe']
    from v2.probe.v1_features import RUNS, records
    labels={(r['id'],r['strategy']):int(r['correct']) for run,_,_ in RUNS for r in records(run)}
    selection=[i for i in items if i['split'] in ('v1_train','v1_validation')]
    for target,block in report['targets'].items():
        strategy='decomposed' if target=='y_decomposed' else 'direct'
        y=[labels[(i['id'],strategy)] for i in selection]
        for row in block['table'].values():
            assert roc_auc_score(y,row['repeat_averaged_oof_scores'])==row['averaged_oof_auroc']
    for target,rows in report['audit']['targets'].items():
        strategy='decomposed' if target=='y_decomposed' else 'direct'
        for key,row in rows.items():
            y=[labels[(i,strategy)] for i in row['ids']]
            assert roc_auc_score(y,row['scores'])==row['auroc']
    result=dict(status='PASS',feature_sets=output,prompt_wrapper_models=4,
                selected=report['selected_layer'],agreement_counts_verified=True,audit_auroc_verified=True)
    (DEST/'readback.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
    print('PASS: five feature sets, prompt wrappers, agreement counts, CV selection, audit AUROCs')


if __name__=='__main__':
    main()
