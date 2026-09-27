"""Reentrant cross-process locks for local stores on macOS/Linux."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import fcntl
import os
import threading

_locks = {}
_guard = threading.Lock()
_local = threading.local()


@contextmanager
def store_lock(path):
    key = str(Path(path).resolve())
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        held = getattr(_local, 'held', {})
        _local.held = held
        if key in held:
            yield
            return
        lock_path = Path(key + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held[key] = fd
            try:
                yield
            finally:
                del held[key]
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def locked(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        store = getattr(self, 'store', self)
        with store_lock(store.path):
            return method(self, *args, **kwargs)
    return invoke
