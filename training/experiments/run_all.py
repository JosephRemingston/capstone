"""Run the three-dataset retention experiment and write the tracked summary.

    python -m training.experiments.run_all                              # every dataset, every stage
    python -m training.experiments.run_all --datasets msc --dry-run
    python -m training.experiments.run_all --skip-build --skip-xgboost --skip-neural --skip-replay   # recombine + report

Stage 0 (dataset build) runs in this process; XGBoost, the neural heads, the artifact replay and the combine step
each run as their own process (xgboost and torch must not share a process on macOS). A failed step is logged under
<out>/<dataset>/logs/ and later steps still run; the report marks what is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from training.experiments import datasets, report

ROOT = datasets.ROOT
DEFAULT_OUT = ROOT / "artifacts" / "experiments" / "retention_datasets"
SHIPPED = {"xgboost": ("importance",), "neural": ("importance_neural", "importance_neural_shallow")}


def steps(name: str, dataset_dir: Path, args) -> list[tuple[str, list[str]]]:
    plan = []
    module = lambda mod, *rest: [sys.executable, "-m", f"training.experiments.{mod}", *rest]
    if not args.skip_xgboost:
        plan.append(("xgboost", module("xgboost_run", "--dataset-dir", str(dataset_dir))))
        if datasets.REGISTRY[name].kind == "csv":
            plan.append(("xgboost_tertile", module("xgboost_run", "--dataset-dir", str(dataset_dir), "--bin-edges", "tertile")))
    if not args.skip_neural:
        plan.append(("neural", module("neural_run", "--dataset-dir", str(dataset_dir))))
    if name == "msc" and not args.skip_replay:
        plan.append(("replay_xgboost", module("replay_shipped", "--model", "xgboost", "--dataset-dir", str(dataset_dir))))
        for artifact in SHIPPED["neural"]:
            plan.append((f"replay_{artifact}", module("replay_shipped", "--model", "neural", "--dir",
                                                      str(ROOT / "artifacts" / artifact), "--dataset-dir", str(dataset_dir))))
    plan.append(("combine", module("combine", "--dataset-dir", str(dataset_dir), "--resamples", str(args.resamples))))
    return plan


def run_step(name: str, command: list[str], log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w") as handle:
        code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    return {"name": name, "command": command[1:], "exit_code": code, "seconds": round(time.perf_counter() - started, 1),
            "log": str(log_path.relative_to(ROOT) if log_path.is_relative_to(ROOT) else log_path)}


def git_state() -> dict:
    # same fields as scripts/measure_all.py:111-119
    def git(*command):
        return subprocess.run(["git", *command], cwd=ROOT, capture_output=True).stdout
    return {"git_head": git("rev-parse", "--short", "HEAD").decode().strip(),
            "dirty_tree": bool(git("status", "--porcelain").strip()),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest()}


def versions() -> dict:
    found = {}
    for package in ("numpy", "scikit-learn", "xgboost", "torch", "fastembed", "onnxruntime", "pyarrow"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def replay_results(dataset_dir: Path) -> dict:
    folder = dataset_dir / "replay"
    return {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))} if folder.is_dir() else {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="msc,csv5k,csv10k")
    parser.add_argument("--csv5k", type=Path, default=None)
    parser.add_argument("--csv10k", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-dir", type=Path, default=report.REPORT_DIR)
    parser.add_argument("--resamples", type=int, default=1000)
    for flag in ("build", "xgboost", "neural", "replay"):
        parser.add_argument(f"--skip-{flag}", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = set(names) - set(datasets.REGISTRY)
    if unknown:
        parser.error(f"unknown datasets: {sorted(unknown)}")
    overrides = {key: value for key, value in (("csv5k", args.csv5k), ("csv10k", args.csv10k)) if value}
    manifest = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "argv": sys.argv,
                **git_state(), "versions": versions(), "datasets": {}, "steps": []}
    results, failed = {}, False
    for name in names:
        dataset_dir = args.out / name
        entry = {"status": "ok", "steps": []}
        if not args.skip_build:
            print(f"[{name}] build", flush=True)
            if not args.dry_run:
                try:
                    datasets.build(name, overrides=overrides, out_dir=args.out)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"[{name}] build failed: {exc}", flush=True)
                    entry.update(status="failed", reason=str(exc))
                    manifest["datasets"][name], results[name], failed = entry, None, True
                    continue
        for step, command in steps(name, dataset_dir, args):
            print(f"[{name}] {step}: {' '.join(command[1:])}", flush=True)
            if args.dry_run:
                continue
            outcome = run_step(f"{name}:{step}", command, dataset_dir / "logs" / f"{step}.log")
            print(f"[{name}] {step}: exit {outcome['exit_code']} in {outcome['seconds']} s", flush=True)
            entry["steps"].append(outcome)
            manifest["steps"].append(outcome)
            failed |= outcome["exit_code"] != 0
        if args.dry_run:
            continue
        combined_path = dataset_dir / "combined.json"
        results[name] = json.loads(combined_path.read_text()) if combined_path.exists() else None
        if results[name] is None:
            entry["status"] = "failed"
        if name == "msc":
            manifest["replay"] = replay_results(dataset_dir)
        manifest["datasets"][name] = entry
    if args.dry_run:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_report:
        json_path, md_path = report.write(report.build_summary(results, manifest), args.report_dir)
        print(f"Report: {md_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
