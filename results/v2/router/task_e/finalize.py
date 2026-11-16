"""Final independent evidence/readback and task delivery record. No fitting except v1 regression."""
import hashlib
import json
import lzma
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from v2.router.readback_p4 import verify as readback
from v2.router.run_final import run as run_final
from v2.router.v1_repro import reproduce
from v2.router.workflow import sha256

BASE = ROOT / 'results/v2/router'
DEST = BASE / 'task_e'
tests = json.loads((DEST / 'tests_result.json').read_text())
test_log = (DEST / 'tests.log').read_text(encoding='utf-8')
assert tests['returncode'] == 0 and re.search(r'Ran 57 tests.*\n\nOK\s*$', test_log, re.S)
tests.update(n_tests=57, skipped=0, log_sha256=sha256(DEST / 'tests.log'))
assert not list(BASE.glob('test_p4_*')), 'Acceptance temporary directory remains'
for domain in ['scan', 'chainops']:
    assert {p.name for p in (BASE / ('mode_a_' + domain)).iterdir()} == {'summary.md'}
with threadpool_limits(limits=1):
    v1 = reproduce()
    assert v1['match']
    readbacks = {name: readback(directory) for name, directory in [
        ('v1', BASE / 'p4-v1adapter'), ('synthetic', BASE / 'synthetic16/p4')]}
code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'v2/router').glob('*.py'))}
for directory in [BASE / 'p4-v1adapter', BASE / 'synthetic16/p4']:
    report = json.loads((directory / 'report.json').read_text())
    assert report['code_sha256'] == code
    for filename, digest in report['frozen_sha256'].items():
        assert sha256(directory / 'frozen' / filename) == digest
    for path, digest in report['input_sha256'].items():
        assert sha256(path) == digest
    for image in (directory / 'figures').glob('*.png'):
        assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert len(list((directory / 'figures').glob('*.png'))) == 4
    assert len((directory / 'summary.md').read_text().splitlines()) < 150
packaging = json.loads((DEST / 'packaging.json').read_text())
for record in packaging['changes']:
    directory = BASE / ('p4-v1adapter' if record['name'] == 'v1' else 'synthetic16/p4')
    payload = lzma.decompress((directory / 'frozen/objects.joblib').read_bytes())
    assert hashlib.sha256(payload).hexdigest() == record['identical_serialized_payload_sha256']
corrections = json.loads((DEST / 'statistics_correction.json').read_text())
curve_points, corrected_auc_values = 0, 0
for record in corrections['records']:
    old = json.loads(lzma.decompress((ROOT / record['original_report_archive']).read_bytes()))
    new = json.loads((ROOT / record['report']).read_text())
    assert old['evaluation_call'] == new['evaluation_call']
    for name, values in old['policies'].items():
        current = new['policies'][name]
        assert len(values['curve']) == len(current['curve'])
        for before, after in zip(values['curve'], current['curve']):
            np.testing.assert_allclose([before['accuracy'], before['mean_cost']],
                                       [after['accuracy'], after['mean_cost']], atol=1e-12, rtol=0)
            curve_points += 1
        if values['normalized_auc']['value'] is None:
            corrected_auc_values += 1
        assert current['normalized_auc']['value'] is not None
refused = []
for split in ['holdout', 'ood_length']:
    try:
        run_final(BASE / 'synthetic16/p4', BASE / 'synthetic16/labels.csv', BASE / 'synthetic16/items',
                  split, BASE / 'synthetic16/features', synthetic_test=True)
    except RuntimeError as error:
        assert 'already completed' in str(error)
        refused.append(split)
    else:
        raise AssertionError('Repeated final evaluation was accepted')
paths = dict(v1=BASE / 'p4-v1adapter/report.json', synthetic=BASE / 'synthetic16/p4/report.json',
             synthetic_holdout=BASE / 'synthetic16/p4/final_holdout/report.json',
             synthetic_ood=BASE / 'synthetic16/p4/final_ood_length/report.json')
numbers = {}
for name, path in paths.items():
    report = json.loads(path.read_text())
    p = report['policies']['selected/overall']
    assert report['uncertainty']['n_boot'] == 10000
    numbers[name] = dict(report=str(path.relative_to(ROOT)), selected_router=report['selected_router'],
                         accuracy=p['accuracy'], mean_cost=p['mean_cost'], mean_overhead=p['mean_overhead'],
                         auc=p['normalized_auc'], best_single=report['best_single'],
                         paired=report['comparisons']['best_single - selected/overall'],
                         fixed_frontier_auc=report['fixed_frontier_auc'], wall_seconds=report['wall_seconds'])
    if report.get('p4_completed'):
        assert not report['quick'] and report['lc_repeats'] == 20
diff = subprocess.run(['git', 'diff', '--check', '--', 'v2/router', 'v2/tests'], cwd=ROOT, capture_output=True, text=True)
assert diff.returncode == 0, diff.stdout + diff.stderr
status = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
(DEST / 'git_status.txt').write_text(status, encoding='utf-8')
timings = json.loads((DEST / 'timings.json').read_text())
full_times = {name: timings[name]['wall_seconds'] for name in ['full_v1', 'full_synthetic', 'full_synthetic_final']}
result = dict(complete=True, tests=tests, v1_repro_match=True, readbacks=readbacks,
              code_and_input_and_frozen_hashes_verified=True, identical_payloads_verified=True,
              final_repeats_refused=refused, results=numbers, full_run_wall_seconds=full_times,
              statistics_correction_seconds=sum(r['wall_seconds'] for r in corrections['records']),
              unchanged_selected_decision_rows=sum(r['unchanged_selected_decision_rows'] for r in corrections['records']),
              unchanged_aggregate_curve_points=curve_points, corrected_missing_auc_values=corrected_auc_values,
              corrected_mode_a_timings=json.loads((DEST / 'corrected_mode_a_timings.json').read_text()),
              llm_calls=0, gpu_used=False, real_final_items_opened=False, git_mutations=False,
              git_status=status, historical_attempt_log='timings.json')
lines = ['# Task E completed', '',
         'Implemented and validated the synthetic fixture generator, P4 development driver, frozen P5 driver,',
         'cross-fitted calibration, batched bootstrap, independent readback, and acceptance tests.',
         'Task D work was preserved. No GPU, Ollama, network, installs, Git mutation, or real v2 final-item reads.',
         '', '## Verified results', '',
         '| Data / split | Selected router | Accuracy | Tokens | Normalized AUC |',
         '|---|---|---:|---:|---:|']
for key, label in [('v1', 'v1 validation'), ('synthetic', 'Synthetic validation'),
                   ('synthetic_holdout', 'Synthetic holdout'), ('synthetic_ood', 'Synthetic OOD length')]:
    r = numbers[key]
    lines.append(f"| {label} | {r['selected_router']} | {r['accuracy']:.4f} | {r['mean_cost']:.3f} | {r['auc']['value']:.6f} |")
lines += ['', 'Validation was used for selection and is optimistic. Synthetic results are fixtures, not capability evidence.',
          'The v1 best single reached 0.3125 accuracy at 688.03125 tokens; selected routing used 601.625 tokens.',
          '', '## Validation and timing', '',
          f"- 57 tests passed, zero skipped, Python exit code 0; baseline was 50 passing tests.",
          '- Historical `v1_repro`: `match: true` (15/32 audit, 625.09375 tokens).',
          '- Independent readback passed for both P4 runs and all synthetic final/S2 reports.',
          '- Source/input/frozen hashes and identical recompressed payloads verified; both final repeats refused.',
          '- Both Mode-A domains reran with the correction; only their summaries remain.',
          f"- Full command wall times: v1 P4 {full_times['full_v1']:.2f}s; synthetic P4 {full_times['full_synthetic']:.2f}s; synthetic final {full_times['full_synthetic_final']:.2f}s.",
          f"- Additional statistics-only correction: {result['statistics_correction_seconds']:.2f}s; no execution-model refits or policy reselection.",
          '', '## Corrections and provenance', '',
          'Independent readback exposed floating-point endpoint ties incorrectly producing null AUCs.',
          'The batched metric now handles numerical ties, with a dedicated regression test and end-to-end readback test.',
          f"All {result['unchanged_selected_decision_rows']:,} selected decision rows and {curve_points:,} aggregate curve points stayed unchanged.",
          f"Corrected {corrected_auc_values} missing observed AUC values; genuine unreachable-target CIs remain undefined.",
          'Original reports and affected sources are archived in `pre_metric_fix/`. `fit_code_sha256` preserves',
          'fit-time provenance; `code_sha256` records the corrected frozen evaluation code. The one-shot evaluation',
          'logs still contain one completed call per split. No final evaluation call was repeated to repair statistics.',
          'Validation cross-fit calibrators were recreated from unchanged raw predictor outputs to replay curves;',
          'final policies used the saved full-validation calibrators without fitting.',
          'Frozen joblib files were losslessly transcoded to LZMA; original serialized payload hashes match exactly.',
          'Original report archives and container hashes preserve provenance. JSON whitespace was compacted losslessly.',
          'Summary table escaping was corrected; crowded synthetic figures were rerendered from saved metrics.',
          '', '## Artifacts and limits', '',
          '- [v1 summary](../p4-v1adapter/summary.md), [synthetic summary](../synthetic16/p4/summary.md).',
          '- [Synthetic holdout](../synthetic16/p4/final_holdout/summary.md), [synthetic OOD](../synthetic16/p4/final_ood_length/summary.md).',
          '- [Mode-A SCAN](../mode_a_scan/summary.md), [Mode-A ChainOps](../mode_a_chainops/summary.md).',
          '- `acceptance.json`: numbers, CIs, wall times, hash/readback checks and final size.',
          '- `tests.log`, `statistics_correction.json`, `packaging.json`, `json_compaction.json`: audit evidence.',
          '- [Git status](git_status.txt) includes pre-existing workstream changes; no commits or other Git writes were made.',
          '- Token cost is a replay proxy. HF/Ollama quantization differs. Missing adapter S0 costs use WS2 probe-token counts.',
          '- Bootstrap conditions on fitted models/choices; readback independently checks decisions/costs and aggregate AUC, not classifier fitting or every bootstrap CI.',
          '', 'No unfinished Task E obligations remain.']
(DEST / 'STATUS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
for _ in range(3):
    result['router_bytes'] = sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
    (DEST / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
assert result['router_bytes'] < 50_000_000, result['router_bytes']
print(json.dumps({k: result[k] for k in ['complete', 'tests', 'router_bytes', 'full_run_wall_seconds',
                                      'unchanged_aggregate_curve_points', 'corrected_missing_auc_values']}, indent=2))
print(json.dumps(numbers, indent=2))
