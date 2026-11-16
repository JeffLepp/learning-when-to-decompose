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


@contextmanager
def gpu_lock(purpose, poll=5.0):
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
        current = holder()
        if current and current.get('pid') == os.getpid():
            LOCK.unlink(missing_ok=True)
