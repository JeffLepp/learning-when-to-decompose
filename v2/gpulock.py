"""Cross-process exclusive GPU lock. Every process that runs Ollama inference or
loads a Hugging Face model onto the GPU must hold this lock for the whole job.

Usage:
    from v2.gpulock import gpu_lock
    with gpu_lock('ws1-collect scan router_train qwen3:4b'):
        ...

The lock file records holder PID/purpose/time. A lock whose PID no longer
exists is treated as stale and reclaimed. Waiting is logged every 60 s.
"""
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(__file__).resolve().parent.parent / '.gpu.lock'


def _pid_alive(pid):
    if sys.platform == 'win32':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def holder():
    try:
        return json.loads(LOCK.read_text())
    except (OSError, ValueError):
        return None


def unload_ollama_models():
    """Best effort: evict every resident Ollama model (frees VRAM and the
    llama-server RAM that caused memory pressure on 2026-11-15)."""
    import urllib.request
    api = 'http://127.0.0.1:11434/api/'
    try:
        with urllib.request.urlopen(api + 'ps', timeout=10) as r:
            names = [m['name'] for m in json.loads(r.read()).get('models', [])]
        for name in names:
            req = urllib.request.Request(api + 'generate', data=json.dumps({'model': name, 'keep_alive': 0}).encode(),
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=60).read()
    except Exception as exc:  # never mask the caller's own result
        print(f'[gpulock] ollama unload skipped: {exc}', flush=True)


@contextmanager
def gpu_lock(purpose, poll=5.0, unload_on_release=True):
    info = {'pid': os.getpid(), 'purpose': purpose, 'since': time.strftime('%Y-%m-%dT%H:%M:%S')}
    last_log = 0.0
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as f:
                json.dump(info, f)
            break
        except FileExistsError:
            current = holder()
            if current is None or not _pid_alive(int(current.get('pid', -1))):
                try:
                    LOCK.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.time() - last_log > 60:
                print(f'[gpulock] waiting; held by {current}', flush=True)
                last_log = time.time()
            time.sleep(poll)
    try:
        yield
    finally:
        if unload_on_release:
            unload_ollama_models()
        current = holder()
        if current and current.get('pid') == os.getpid():
            LOCK.unlink(missing_ok=True)
