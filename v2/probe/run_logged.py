"""Capture a probe module as plain ASCII without PowerShell stderr rewriting."""
import os
from pathlib import Path
import subprocess
import sys

module, logname = sys.argv[1:]
assert module in ('check_env', 'match_models', 'verify_matching', 'v1_features', 'feasibility_v1')
assert Path(logname).name == logname and logname.startswith('WS2-')
env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONIOENCODING='utf-8')
with (Path('workspace_logs') / logname).open('a', encoding='ascii') as log:
    log.write('Starting module: ' + module + '\n')
    proc = subprocess.Popen([sys.executable, '-m', 'v2.probe.' + module],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    for line in iter(proc.stdout.readline, b''):
        log.write(line.decode('utf-8', errors='replace').encode('ascii', errors='backslashreplace').decode('ascii'))
        log.flush()
    code = proc.wait()
    log.write('Process exit code: ' + str(code) + '\n')
sys.exit(code)
