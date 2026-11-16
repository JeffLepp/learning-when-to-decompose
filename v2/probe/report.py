"""Render measured WS2 results for the workstream handoff."""
import json
from v2.probe.v1_features import DEST, ROOT


def load(name):
    return json.loads((DEST/name).read_text())


def main():
    assert load('readback.json')['status']=='PASS'
    throughput=load('throughput.json')
    feasibility=load('feasibility_report.json')
    lines=['# WS2 T2 results','',
           'Completed: extraction pipeline, CPU tests, five throughput configurations, teacher-forced agreement, nested-CV feasibility, exploratory audit, independent readback.','',
           'Files: v2/probe/{extract,prompts,v1_features,feasibility_v1,readback,report,test_logged,safe_discovery,call_costs}.py; v2/probe/README.md; v2/probe/prompt_format_check.json; v2/tests/test_probe_extract.py; results/v2/features/scan_v1/; workspace_logs/WS2-*.','',
           '| Model / forward | Load s | Forward pipeline s | Items/s | Tokens/s | Peak GiB | CPU layers | Directory MiB |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,row in throughput.items():
        lines.append(f"| {name} ({row['quantization']}) | {row['load_seconds']:.2f} | {row['wall_seconds']:.2f} | {row['items_per_second']:.3f} | {row['tokens_per_second']:.1f} | {row['cuda_max_memory_allocated']/2**30:.3f} | {row['cpu_layers']} | {row['feature_dir_bytes']/2**20:.3f} |")
    lines.extend(['','Peak is torch.cuda.max_memory_allocated, not total driver/desktop VRAM. Forward pipeline includes tokenization, batching and pooling; model loading, warmup and file saving are excluded. The 8B CPU placement is embedding/output tables, not transformer layers.','',
        'Prompt-ID matches: four live wrappers 4/4 (28/28 tokens for 0.6B, 1.7B, 8B; 22/22 for 4B); v1 8B 184/184 commands; v1 1.7B 24/24 commands. Exactly four one-token nondataset generation checks were made, with full responses/costs saved.','',
        '| Teacher-forced model | Items | All recorded generated tokens | Letter-bearing tokens |',
        '| --- | ---: | ---: | ---: |'])
    missing_notes=[]
    for name,row in load('teacher_forced_agreement.json').items():
        missing=sum(r['recorded_eval_count']-r['positions'] for r in row['records'])
        lines.append(f"| {name} | {row['n_items']} | {row['matches']}/{row['positions']} ({100*row['agreement']:.2f}%) | {row['letter_matches']}/{row['letter_positions']} ({100*row['letter_agreement']:.2f}%) |")
        missing_notes.append(f'{name}: {missing} eval_count tokens absent from stored context are excluded.')
    lines.extend(['']+missing_notes+[''])
    lines.extend(['Ollama generated under a JSON-schema grammar; HF argmax is unconstrained.','',
                  '| 4B layer | Mean per-item cosine, last pooling, BF16 vs NF4 |','| --- | ---: |'])
    for layer,sim in load('4b_comparison.json')['mean_per_item_cosine_last'].items():
        lines.append(f'| {layer} | {sim:.6f} |')
    lines.extend(['','Selection: n=128, train plus validation only. Nested 5-fold x 5-repeat CV; inner 3-fold C choice. CIs use 2000 item bootstrap resamples of repeat-averaged OOF scores.','',
                  '| Target | Feature | Mean CV AUROC | 95% CI |','| --- | --- | ---: | --- |'])
    for target,block in feasibility['targets'].items():
        table=block['table']
        top=sorted((k for k in table if k.startswith('L')),key=lambda k:-table[k]['mean_oof_auroc'])[:3]
        for key in top+['F1','F2']:
            row=table[key]
            lo,hi=row['bootstrap']['ci95']
            lines.append(f"| {target} | {key} | {row['mean_oof_auroc']:.4f} | {lo:.4f}-{hi:.4f} |")
    lines.extend(['','Selected primary probe: '+feasibility['selected_layer']['feature_set'], '',
                  '| Primary selected probe minus baseline | AUROC difference | Paired 95% CI |','| --- | ---: | --- |'])
    for key,row in feasibility['targets']['y_decomposed']['paired_comparisons'].items():
        lo,hi=row['bootstrap']['ci95']
        lines.append(f"| {key} | {row['difference']:.4f} | {lo:.4f} to {hi:.4f} |")
    lines.extend(['','These differences are optimistically biased by choosing the best layer on the same selection items.','',
                  'exploratory: v1 audit was already inspected in v1; n=32','',
                  '| Target | Feature | Audit AUROC | 95% CI |','| --- | --- | ---: | --- |'])
    for target,rows in feasibility['audit']['targets'].items():
        for key,row in rows.items():
            lo,hi=row['bootstrap']['ci95']
            lines.append(f"| {target} | {key} | {row['auroc']:.4f} | {lo:.4f}-{hi:.4f} |")
    lines.extend(['','Label counts: '+json.dumps(feasibility['positives_per_target_per_split'],sort_keys=True),'',
        'Problems / deviations:', '',
        '- Initial 8B NF4 automatic offload failed on meta quantization state. Explicit placement fixed and verified; no partial feature folder was left. Failed load/warmup cost was not captured and is marked unknown in execution_failures.json.',
        '- The successful runs spanned two GPU-owning processes after that failure. Waiting processes were restarted for code fixes/polling. Completed collection and the four Ollama checks were preserved.',
        '- WS1 held the shared GPU for an extended interval. WS2 used the same lock throughout, with supported poll=0.5 for the resumed run.',
        '- qwen3:4b is Thinking-2507 and always renders a thinking prefix, unlike the hybrid sizes. The model ladder is heterogeneous.',
        '- S0 adapter currently selects WS1 domain_v0.json; follow a newer WS1 prompt version when it is selected. No copied S0 text.',
        '- Small n, imbalanced y_direct, NF4 versus Q4_K_M, and previously inspected audit limit the conclusions. No official OOD inference occurred.',
        '- Two unguarded full-suite runs (45 and later 60 tests) invoked existing WS1 tests that read the forbidden official length-test file. This was caught after execution. No model inference or feature selection used that data. Further suite runs use safe_discovery, which prevents that file read and skips the test. Original and guarded logs are preserved separately.',
        '- Teacher-forced wall times and per-batch wall times were not recorded. call_costs.json records input-token accounting, while extraction forward-pipeline wall times are recorded in aggregate.',
        '- This task did not change packages, git history, shared initialization files, HANDOFF.md, or other workstream source files.', '',
        'Test log summaries:',''])
    tests=(ROOT/'workspace_logs/WS2-tests.txt').read_text()
    lines.extend(line for line in tests.splitlines() if line.startswith(('COMMAND:','Ran ','OK','EXIT:')))
    lines.extend(['','Guarded validation:',''])
    tests=(ROOT/'workspace_logs/WS2-tests-guarded.txt').read_text()
    lines.extend(line for line in tests.splitlines() if line.startswith(('COMMAND:','Ran ','OK','EXIT:','Forbidden file reads prevented:')))
    lines.extend(['','Independent readback: PASS.'])
    path=ROOT/'workspace_logs/WS2-T2-status.md'
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    print(path.read_text())


if __name__=='__main__':
    main()
