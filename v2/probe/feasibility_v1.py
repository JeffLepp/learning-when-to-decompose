"""CPU-only nested-CV feasibility analysis on preserved v1 labels."""
import json
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from controller import features
from v2.probe.v1_features import DEST, ROOT, RUNS, records, save, v1_items

CS = [1e-4,1e-3,1e-2,1e-1,1]


def attach_execution_notes(report):
    path=DEST/'execution_failures.json'
    if path.exists():
        report['execution_notes']=json.loads(path.read_text())
        note='Unguarded full-suite tests accessed the forbidden official length-test file; no model inference or feature selection used that file. Subsequent validation blocked those reads.'
        if note not in report['caveats']:
            report['caveats'].append(note)
    return report


def pipeline(kind):
    steps = []
    if kind=='F1':
        steps.append(('vectorizer',DictVectorizer(sparse=False)))
    elif kind=='F2':
        steps.append(('tfidf',TfidfVectorizer(ngram_range=(1,3))))
    if kind!='F2':
        steps.append(('scaler',StandardScaler()))
    # sklearn 1.9: l1_ratio=0 is L2; the old penalty argument is deprecated.
    steps.append(('lr',LogisticRegression(l1_ratio=0,C=1,class_weight=None,
                 max_iter=5000,solver='liblinear',random_state=437)))
    return Pipeline(steps)


def fit_inner(x,y,kind):
    search = GridSearchCV(pipeline(kind),{'lr__C':CS},scoring='roc_auc',
             cv=StratifiedKFold(3,shuffle=True,random_state=437),n_jobs=1,refit=True,error_score='raise')
    search.fit(x,y)
    return search


def bootstrap(y,scores,other=None):
    rng = np.random.default_rng(437)
    values=[]
    for _ in range(2000):
        ids=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ids]))<2:
            continue
        value=roc_auc_score(y[ids],scores[ids])
        if other is not None:
            value-=roc_auc_score(y[ids],other[ids])
        values.append(value)
    return dict(ci95=np.quantile(values,[.025,.975]).tolist(),valid_resamples=len(values),
                resamples=2000,seed=437)


def nested(x,y,kind):
    scores=np.full((5,len(y)),np.nan)
    choices=[]
    splitter=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=437)
    for fold,(train,test) in enumerate(splitter.split(np.zeros(len(y)),y)):
        model=fit_inner(x[train],y[train],kind)
        scores[fold//5,test]=model.predict_proba(x[test])[:,1]
        choices.append(model.best_params_['lr__C'])
    assert np.isfinite(scores).all()
    repeat_auc=[float(roc_auc_score(y,s)) for s in scores]
    averaged=scores.mean(0)
    return dict(mean_oof_auroc=float(np.mean(repeat_auc)),repeat_aurocs=repeat_auc,
                averaged_oof_auroc=float(roc_auc_score(y,averaged)),
                bootstrap=bootstrap(y,averaged),chosen_C_per_outer_fold=choices,
                repeat_averaged_oof_scores=averaged.tolist())


def main():
    # Limit CPU contention with the other workstreams; BLAS is not needed here.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run()


def run():
    if (DEST/'feasibility_report.json').exists():
        raise FileExistsError('Final feasibility report already exists; do not repeat the audit evaluation.')
    items=v1_items()
    index=json.loads((DEST/'qwen3-8b/index.json').read_text())
    assert index['ids']==[i['id'] for i in items]
    lookup={}
    counts={}
    for run,split,n in RUNS:
        rows=records(run)
        for row in rows:
            lookup[(row['id'],row['strategy'])]=int(row['correct'])
        counts[split]=dict(n=n,**{target:sum(lookup[(i['id'],strategy)] for i in items if i['split']==split)
              for target,strategy in [('y_decomposed','decomposed'),('y_direct','direct')]})
    selection=np.array([i for i,item in enumerate(items) if item['split'] in ('v1_train','v1_validation')])
    audit=np.array([i for i,item in enumerate(items) if item['split']=='v1_audit'])
    assert len(selection)==128 and len(audit)==32
    all_sets={}
    for layer in index['layers']:
        for pooling in index['poolings']:
            key=f'L{layer:02d}_{pooling}'
            values=np.load(DEST/'qwen3-8b'/index['files'][key]).astype(np.float32)
            assert values.shape==(184,index['hidden_size'])
            all_sets[key]=(values,'probe')
    all_sets['F1']=(np.array([features(i['input']) for i in items],dtype=object),'F1')
    all_sets['F2']=(np.array([i['input'] for i in items],dtype=object),'F2')
    caveats=['Feasibility only: n=128 selection items.',
        'HF 8B features use 4-bit NF4 weights; Ollama labels use Q4_K_M.',
        'All correctness labels come from preserved Ollama runs.',
        'Selecting the best layer on these same 128 items makes paired AUROC differences optimistically biased.',
        'Bootstrap intervals resample items with repeat-averaged out-of-fold scores; they do not refit models or repeat layer selection.',
        'Secondary y_direct has very few positives.',
        'v1_dev is excluded from fitting and selection; only its label counts are reported.',
        'Exploratory audit: v1 audit was already inspected in v1; n=32.']
    report=dict(n=128,positives_per_target_per_split=counts,caveats=caveats,
        protocol=dict(outer='RepeatedStratifiedKFold(5,5,random_state=437)',
            inner='StratifiedKFold(3,shuffle=True,random_state=437)',C=CS,
            fitting='All scaling, vocabulary/IDF, and C selection fitted inside each outer training fold.',
            bootstrap='2000 item resamples, seed 437, skip single-class samples'),targets={})
    for target,strategy in [('y_decomposed','decomposed'),('y_direct','direct')]:
        y=np.array([lookup[(i['id'],strategy)] for i in items])
        table={}
        for key,(x,kind) in all_sets.items():
            table[key]=nested(x[selection],y[selection],kind)
            print(target,key,table[key]['mean_oof_auroc'],table[key]['bootstrap']['ci95'],flush=True)
        best=max((k for k in table if k.startswith('L')),key=lambda k:table[k]['mean_oof_auroc'])
        comparison={}
        for baseline in ['F1','F2']:
            a=np.array(table[best]['repeat_averaged_oof_scores'])
            b=np.array(table[baseline]['repeat_averaged_oof_scores'])
            comparison[baseline]=dict(difference=float(roc_auc_score(y[selection],a)-roc_auc_score(y[selection],b)),
                bootstrap=bootstrap(y[selection],a,b),optimistically_biased=True)
        report['targets'][target]=dict(table=table,cv_best_probe=best,paired_comparisons=comparison)
        save(DEST/'feasibility_report.partial.json',report)
    best=report['targets']['y_decomposed']['cv_best_probe']
    report['selected_layer']=dict(feature_set=best,layer=int(best[1:3]),pooling=best.split('_',1)[1],
                                  target='y_decomposed',criterion='highest mean repeated OOF AUROC')
    # Audit is evaluated once, only after all primary selection is complete.
    audit_results={}
    for target,strategy in [('y_decomposed','decomposed'),('y_direct','direct')]:
        y=np.array([lookup[(i['id'],strategy)] for i in items])
        audit_results[target]={}
        for key in [best,'F1','F2']:
            x,kind=all_sets[key]
            model=fit_inner(x[selection],y[selection],kind)
            scores=model.predict_proba(x[audit])[:,1]
            audit_results[target][key]=dict(auroc=float(roc_auc_score(y[audit],scores)),
                bootstrap=bootstrap(y[audit],scores),C=model.best_params_['lr__C'],
                ids=[items[i]['id'] for i in audit],scores=scores.tolist())
    report['audit']=dict(label='exploratory: v1 audit was already inspected in v1; n=32',
                         n=32,probe_selected_on='y_decomposed selection CV only',targets=audit_results)
    attach_execution_notes(report)
    save(DEST/'feasibility_report.json',report)
    (DEST/'feasibility_report.partial.json').unlink(missing_ok=True)
    lines=['# V1 activation feasibility','', 'Selection n=128; nested 5-fold x 5-repeat CV; 95% item bootstrap CIs.','',
           '| Target | Feature | CV AUROC | 95% CI |','| --- | --- | ---: | --- |']
    for target,block in report['targets'].items():
        for key,row in sorted(block['table'].items(),key=lambda kv:-kv[1]['mean_oof_auroc']):
            lo,hi=row['bootstrap']['ci95']
            lines.append(f"| {target} | {key} | {row['mean_oof_auroc']:.4f} | {lo:.4f}-{hi:.4f} |")
    lines.extend(['','Selected primary probe: '+best,'','exploratory: v1 audit was already inspected in v1; n=32','',
                  '| Target | Feature | Audit AUROC | 95% CI |','| --- | --- | ---: | --- |'])
    for target,rows in audit_results.items():
        for key,row in rows.items():
            lo,hi=row['bootstrap']['ci95']
            lines.append(f"| {target} | {key} | {row['auroc']:.4f} | {lo:.4f}-{hi:.4f} |")
    lines.extend(['']+['- '+c for c in caveats])
    (ROOT/'workspace_logs/WS2-feasibility.md').write_text('\n'.join(lines)+'\n',encoding='ascii')
    print('COMPLETE',DEST/'feasibility_report.json',flush=True)


if __name__=='__main__':
    main()
