"""Train the default importance model end to end.

    python -m training.ensemble.run                      # build, XGBoost, deep heads, assemble
    python -m training.ensemble.run --csv ~/Downloads/cognimem_memory_importance_5000.csv
    python -m training.ensemble.run --dry-run

Stage 0 (dataset build) runs in this process; the XGBoost stage, the deep-head stage and the assemble step
each run as their own process (xgboost and torch must not share a process on macOS). Logs go to
``<out>/logs/``; the run stops at the first failed step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from training.ensemble import datasets

ROOT = datasets.ROOT


def steps(out_dir: Path) -> list[tuple[str, list[str]]]:
    module = lambda mod, *rest: [sys.executable, "-m", f"training.ensemble.{mod}", "--out", str(out_dir), *rest]
    return [("xgboost", module("xgboost_stage")), ("neural", module("neural_stage")), ("assemble", module("assemble"))]


def git_state() -> dict:
    def git(*command):
        return subprocess.run(["git", *command], cwd=ROOT, capture_output=True).stdout
    return {"git_head": git("rev-parse", "--short", "HEAD").decode().strip(),
            "dirty_tree": bool(git("status", "--porcelain").strip()),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=None, help="source CSV (default data/experiments/csv5k.csv)")
    parser.add_argument("--out", type=Path, default=datasets.DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "argv": sys.argv,
                **git_state(), "steps": []}
    print("[build] csv5k", flush=True)
    if not args.dry_run:
        datasets.build(out_dir=args.out, csv_path=args.csv)
    for name, command in steps(args.out):
        print(f"[{name}] {' '.join(command[1:])}", flush=True)
        if args.dry_run:
            continue
        log = args.out / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log.open("w") as handle:
            code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
        seconds = round(time.perf_counter() - started, 1)
        manifest["steps"].append({"name": name, "exit_code": code, "seconds": seconds, "log": str(log)})
        print(f"[{name}] exit {code} in {seconds} s", flush=True)
        if code != 0:
            print(log.read_text()[-3000:], file=sys.stderr)
            break
    if not args.dry_run:
        (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 1 if any(step["exit_code"] for step in manifest["steps"]) else 0


if __name__ == "__main__":
    sys.exit(main())
