"""Persistent cleanup worker. Run under a process supervisor or use --once in cron."""
import argparse
import json
import math
import sqlite3
import signal
import threading
from .cleanup import cleanup
from .store import LocalMemoryStore


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--store', default='memory_store/memories.jsonl')
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--grace-days', type=float, default=7)
    parser.add_argument('--interval-hours', type=float, default=24)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if (not math.isfinite(args.interval_hours) or not math.isfinite(args.grace_days)
            or args.interval_hours <= 0 or args.grace_days < 0):
        parser.error('Invalid cleanup interval or grace period')
    stop = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stop.set())
    while not stop.is_set():
        try:
            result = cleanup(LocalMemoryStore(args.store), user_id=args.user_id,
                             grace_days=args.grace_days, interval_hours=args.interval_hours,
                             scheduled=True, apply=True)
            print(json.dumps(result, sort_keys=True), flush=True)
        except (ValueError, OSError, sqlite3.Error) as exc:
            # Do not advance schedule state after errors; next tick retries.
            print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        stop.wait(min(60, args.interval_hours * 3600))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
