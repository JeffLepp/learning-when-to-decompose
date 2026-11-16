"""Derive forward-call token accounting from saved, measured artifacts."""
import json
from v2.probe.v1_features import DEST, ROOT, records, save


def main():
    throughput=json.loads((DEST/'throughput.json').read_text())
    manifest=dict(feature_extraction={},teacher_forced={},
        prompt_checks='v2/probe/prompt_format_check.json',
        failures='results/v2/features/scan_v1/execution_failures.json',
        caveat='Forward input-token counts measure processing cost, not generated tokens or FLOPs. Per-batch wall times and teacher-forced wall times were not recorded; extraction wall time is available in aggregate.')
    for folder,summary in throughput.items():
        index=json.loads((DEST/folder/'index.json').read_text())
        order=sorted(range(len(index['ids'])),key=lambda i:index['n_tokens'][i])
        size=index['batch_size']
        batches=[]
        for start in range(0,len(order),size):
            group=order[start:start+size]
            batches.append(dict(ids=[index['ids'][i] for i in group],
                real_input_tokens=sum(index['n_tokens'][i] for i in group),
                padded_input_tokens=len(group)*max(index['n_tokens'][i] for i in group)))
        warmup=list(range(min(size,len(index['ids']))))
        manifest['feature_extraction'][folder]=dict(forward_calls=len(batches),
            warmup_calls=1,warmup_ids=[index['ids'][i] for i in warmup],
            warmup_real_input_tokens=sum(index['n_tokens'][i] for i in warmup),
            real_input_tokens=index['total_prompt_tokens'],batches=batches,
            wall_seconds_excluding_warmup=index['wall_seconds'],load_seconds=index['load_seconds'])
    for model,run in [('qwen3:8b','train-8b-v1'),('qwen3:1.7b','pilot-v1')]:
        calls=[]
        for row in records(run):
            if row['strategy']!='direct':
                continue
            call=row['calls'][0]
            calls.append(dict(id=row['id'],input_tokens=len(call['context']),
                prompt_tokens=call['prompt_eval_count'],
                recorded_generated_tokens=len(call['context'])-call['prompt_eval_count']))
        manifest['teacher_forced'][model]=dict(forward_calls=len(calls),
            total_input_tokens=sum(c['input_tokens'] for c in calls),calls=calls,
            wall_seconds=None)
    save(DEST/'call_costs.json',manifest)
    print('Saved forward-call token accounting for five feature sets and two teacher-forced checks.')


if __name__=='__main__':
    main()
