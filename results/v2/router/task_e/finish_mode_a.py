import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'results/v2/router'
DEST = BASE / 'task_e'
timings = {}
for domain in ['scan', 'chainops']:
    directory = BASE / ('mode_a_' + domain)
    shutil.copyfile(directory / 'summary.md', DEST / 'pre_metric_fix' / f'mode_a_{domain}_summary.md')
    commands = [
        ('generate', ['v2.router.synth_labels', '--domain', domain, '--out', str(directory / 'labels_corrected.csv'), '--seed', '437']),
        ('p4', ['v2.router.run_p4', '--labels', str(directory / 'labels_corrected.csv'), '--domain', domain,
                '--items-dir', str(ROOT / 'data/v2' / domain), '--out', str(directory / 'p4_corrected'), '--quick'])]
    for stage, args in commands:
        started = time.perf_counter()
        key = f'corrected_mode_a_{domain}_{stage}'
        print('Starting ' + key, flush=True)
        with (DEST / (key + '.log')).open('w', encoding='utf-8') as stream:
            result = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-B', '-m', *args], cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT)
        timings[key] = dict(wall_seconds=time.perf_counter() - started, returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(key + ' failed')
        print(key + ' completed', flush=True)
    shutil.copyfile(directory / 'p4_corrected/summary.md', directory / 'summary.md')
    resolved = directory.resolve()
    assert resolved.parent == BASE.resolve() and resolved.name == 'mode_a_' + domain
    for child in directory.iterdir():
        assert child.resolve().is_relative_to(resolved)
        if child.name != 'summary.md':
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
(DEST / 'corrected_mode_a_timings.json').write_text(json.dumps(timings, indent=2) + '\n')

