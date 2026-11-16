"""Capture unittest stderr in Python so Windows PowerShell preserves exit status."""
import subprocess
import sys
import time
from pathlib import Path

started = time.perf_counter()
log = Path(__file__).with_name('taskF-tests-verified.log')
arguments = sys.argv[1:] or ['discover', '-s', 'v2/tests', '-p', 'test_router_*.py', '-t', '.']
with log.open('w', encoding='utf-8') as stream:
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', *arguments],
                            stdout=stream, stderr=subprocess.STDOUT)
print(f'Unittest exit={result.returncode}; wall={time.perf_counter() - started:.2f}s; log={log}')
sys.exit(result.returncode)
