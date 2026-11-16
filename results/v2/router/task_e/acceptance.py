"""Task E's requested persistent acceptance runs. CPU only; no real final data."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'results/v2/router'
DEST = BASE / 'task_e'
timings = {}


def call(name, module, *arguments):
    command = [str(ROOT / '.venv/Scripts/python.exe'), '-B', '-m', module, *map(str, arguments)]
    print(f'Starting {name}', flush=True)
    start = time.perf_counter()
    with (DEST / f'{name}.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    timings[name] = dict(wall_seconds=time.perf_counter() - start, command=command,
                         returncode=completed.returncode)
    (DEST / 'timings.json').write_text(json.dumps(timings, indent=2) + '\n', encoding='utf-8')
    print(f'Finished {name}: {timings[name]["wall_seconds"]:.2f}s, exit {completed.returncode}', flush=True)
    if completed.returncode:
        raise RuntimeError(f'{name} failed; see {DEST / (name + ".log")}')


call('full_v1', 'v2.router.run_p4', '--labels', BASE / 'v1_adapter/scan_v1_labels.csv',
     '--domain', 'scan', '--items-dir', ROOT / 'data/v2/scan', '--out', BASE / 'p4-v1adapter',
     '--features-dir', ROOT / 'results/v2/features/scan_v1', '--expected-arms', 'none')
synth = BASE / 'synthetic16'
call('generate_mode_b', 'v2.router.synth_labels', '--domain', 'scan', '--out', synth / 'labels.csv',
     '--synthetic-items-out', synth / 'items', '--features-out', synth / 'features', '--seed', '437')
call('full_synthetic', 'v2.router.run_p4', '--labels', synth / 'labels.csv', '--domain', 'scan',
     '--items-dir', synth / 'items', '--out', synth / 'p4', '--features-dir', synth / 'features')
call('full_synthetic_final', 'v2.router.run_final', '--p4-dir', synth / 'p4', '--labels', synth / 'labels.csv',
     '--items-dir', synth / 'items', '--splits', 'holdout,ood_length',
     '--features-dir', synth / 'features', '--synthetic-test')
for domain in ['scan', 'chainops']:
    directory = BASE / ('mode_a_' + domain)
    call('generate_mode_a_' + domain, 'v2.router.synth_labels', '--domain', domain,
         '--out', directory / 'labels.csv', '--seed', '437')
    call('quick_mode_a_' + domain, 'v2.router.run_p4', '--labels', directory / 'labels.csv', '--domain', domain,
         '--items-dir', ROOT / 'data/v2' / domain, '--out', directory / 'p4', '--quick')
    shutil.copyfile(directory / 'p4/summary.md', directory / 'summary.md')
    # Delete only these task-created Mode-A outputs, after verifying resolved
    # targets are descendants of the explicit results/v2/router directory.
    resolved = directory.resolve()
    assert resolved.parent == BASE.resolve() and resolved.name == 'mode_a_' + domain
    for child in directory.iterdir():
        assert child.resolve().is_relative_to(resolved)
        if child.name != 'summary.md':
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
call('readback_v1', 'v2.router.readback_p4', '--p4-dir', BASE / 'p4-v1adapter')
call('readback_synthetic', 'v2.router.readback_p4', '--p4-dir', synth / 'p4')
call('v1_repro', 'v2.router.v1_repro')
call('tests', 'unittest', 'discover', '-s', 'v2/tests', '-p', 'test_router_*.py', '-t', '.')
paths = dict(v1=BASE / 'p4-v1adapter/report.json', synthetic=synth / 'p4/report.json',
             synthetic_holdout=synth / 'p4/final_holdout/report.json',
             synthetic_ood=synth / 'p4/final_ood_length/report.json')
numbers = {}
for name, path in paths.items():
    report = json.loads(path.read_text(encoding='utf-8'))
    selected = report['policies']['selected/overall']
    numbers[name] = dict(report=str(path.relative_to(ROOT)), selected_router=report['selected_router'],
                         accuracy=selected['accuracy'], mean_cost=selected['mean_cost'],
                         overhead=selected['mean_overhead'], auc=selected['normalized_auc'],
                         best_single=report['best_single'], fixed_frontier_auc=report['fixed_frontier_auc'],
                         paired=report['comparisons']['best_single - selected/overall'],
                         wall_seconds=report['wall_seconds'], n_boot=report['uncertainty']['n_boot'])
status = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
(DEST / 'git_status.txt').write_text(status, encoding='utf-8')
result = dict(timings=timings, results=numbers,
              tests=dict(returncode=timings['tests']['returncode'], log='results/v2/router/task_e/tests.log'),
              v1_repro_match=json.loads((BASE / 'v1_repro/report.json').read_text())['match'],
              router_bytes=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()),
              git_status=status, model_calls=0)
(DEST / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
