"""Tracked report of the default importance model: ``evaluation/reports/retention_ensemble.{json,md}``.

Every number comes from ``assemble.run``; nothing is recomputed or typed by hand.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "evaluation" / "reports"
REPORT_NAME = "retention_ensemble"
CAVEATS = (
    "csv5k is AI-generated: template sentences with synthetic importance scores, not human ratings.",
    "The test split has 1,008 rows but only 86 distinct sentences; the sentence-bootstrap interval is the honest one.",
    "The 0.4 / 0.7 bin edges are a modelling choice.",
    "csv5k sentences are short (at most 13 words); conversation turns are much longer, so serving runs far outside "
    "the training distribution. The system's 25 % model weight is a fixed choice, not tuned; this is a reason to keep it small.",
    "Validation chose the temperatures, blend weight and class weights; only the test numbers are unbiased.",
)


def num(value, digits=4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def ci(interval, digits=3) -> str:
    return "—" if not interval else f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def render_markdown(summary: dict) -> str:
    dataset, diag = summary["dataset"], summary["dataset"]["diagnostics"]
    names = summary["class_names"]
    tuned, model = summary["tuned"], summary["ensemble_thresholds"]
    lines = [
        "# Default importance model: csv5k Ensemble + thresholds",
        "",
        f"Generated {summary['generated_at']} by `python -m training.ensemble.run`. "
        f"The serving artifact is `{summary['artifact']['dir']}/` (metadata sha256 `{summary['artifact']['metadata_sha256'][:12]}`).",
        "",
        f"{summary['model']}. The system's default importance score is 25 % this model's class value plus 75 % the "
        "heuristic score (`main/ensemble.py`).",
        "",
        "## Dataset",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Source | `{dataset['source']['path']}` (sha256 `{dataset['source']['sha256'][:12]}`) |",
        f"| Rows / distinct sentences | {diag['rows']:,} / {diag['unique_sentences']:,} |",
        f"| Words per sentence | mean {diag['words_per_text']['mean']:.1f}, max {diag['words_per_text']['max']} |",
        f"| Importance score | {diag['score_range'][0]:.2f}–{diag['score_range'][1]:.2f} (mean {diag['score_mean']:.3f}) |",
        f"| Classes | low < {dataset['bin_edges'][0]} ≤ mid < {dataset['bin_edges'][1]} ≤ high |",
        f"| Split | {dataset['split_protocol']}, seed {dataset['seed']} |",
        "| Train / validation / test rows | " + " / ".join(f"{dataset['split_sizes'][k]:,}" for k in ("fit", "val", "test")) + " |",
        "| Train / validation / test sentences | " + " / ".join(f"{dataset['split_sentences'][k]:,}" for k in ("fit", "val", "test")) + " |",
        "| Test class counts | " + " / ".join(f"{names[i]} {dataset['class_counts']['test'][n]}" for i, n in enumerate(names)) + " |",
        f"| Repeated sentences whose labels disagree | {diag['disagreeing_multi_groups']} of {diag['multi_row_groups']} |",
        "",
        "## Test results",
        "",
        "Macro F1 intervals are 95% bootstrap intervals over test rows and over whole test sentences "
        f"({summary['resamples']:,} resamples, seed {summary['bootstrap_seed']}).",
        "",
        "| Model | Accuracy | Balanced accuracy | Macro F1 | Row CI | Sentence CI | Weighted F1 | F1 " + " / ".join(names) + " |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for label, item in (("Always the most common class (" + summary["majority"]["class_name"] + ")", summary["majority"]),
                        ("Best possible (oracle)", summary["oracle"]),
                        ("**Ensemble + thresholds**", model)):
        test = item["test"]
        lines.append(f"| {label} | {pct(test['accuracy'])} | {pct(test['balanced_accuracy'])} | {num(test['macro_f1'])} | "
                     f"{ci(item['row_ci95'])} | {ci(item['sentence_ci95'])} | {num(test['weighted_f1'])} | "
                     + " / ".join(f"{test['per_class'][n]['f1']:.2f}" for n in names) + " |")
    matrix = model["test"]["confusion_matrix"]
    calibration = model["probability_calibration"]
    lines += [
        "",
        "The oracle predicts each test sentence's most common test label: a ceiling that reads the test labels, not a model.",
        "",
        "Confusion matrix (rows gold, columns predicted):",
        "",
        "| Gold \\ predicted | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
        *[f"| {names[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(matrix)],
        "",
        f"Validation macro F1 (optimistic, used for tuning): {num(model['validation_macro_f1'])}. "
        f"Blended probabilities before the class weights: test NLL {num(calibration['test_nll'])}, "
        f"ECE {num(calibration['test_ece_10_bins'])}.",
        "",
        "## Model",
        "",
        "| Part | Value |",
        "| --- | --- |",
        f"| XGBoost member | 525 features; max depth {summary['xgboost']['chosen']['max_depth']}, min child weight "
        f"{summary['xgboost']['chosen']['min_child_weight']}, {summary['xgboost']['rounds']} rounds (fit split only); "
        f"temperature {num(tuned['xgb_temperature'])} |",
        f"| Deep head | {summary['neural']['architecture']} on {summary['neural']['encoder_id']}; seeds "
        + ", ".join(f"{s['seed']} (epoch {s['best_epoch']}, T {num(tuned['per_seed_temperature'][str(s['seed'])])})"
                    for s in summary["neural"]["seeds"])
        + f"; seed-average temperature {num(tuned['ensemble_temperature'])} |",
        f"| Blend | {tuned['w_xgb']} × XGBoost + {1 - tuned['w_xgb']} × deep head (validation: "
        + ", ".join(f"w {row['w']} → {num(row['macro_f1'])}" for row in tuned["blend_table"]) + ") |",
        f"| Class weights | {' / '.join(f'{w:g}' for w in tuned['class_weights'])} (validation macro F1 "
        f"{num(tuned['val_macro_f1_before'])} → {num(tuned['val_macro_f1_after'])}) |",
        f"| Class values (serving score) | " + ", ".join(f"{n} {v}" for n, v in zip(names, summary["class_values"]))
        + " (mean csv5k score of the fit + validation rows in each class; the test split is excluded) |",
        "",
        "## Serving checks",
        "",
        "| Check | Rows | Class mismatches | Max probability difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (("onnx_val", "ONNX + saved XGBoost, validation (training embeddings)"),
                       ("onnx_test", "ONNX + saved XGBoost, test (training embeddings)"),
                       ("live_test", "Live single-text BGE embeddings, test")):
        check = summary["serving_checks"][key]
        rows = check.get("rows", dataset["split_sizes"]["val" if key == "onnx_val" else "test"])
        lines.append(f"| {label} | {rows:,} | {check['class_mismatches']} | {check['max_abs_probability_diff']:.2e} |")
    lines += ["", "## Caveats", "", *[f"- {item}" for item in CAVEATS], ""]
    return "\n".join(lines)


def write(summary: dict, report_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **summary,
               "caveats": list(CAVEATS)}
    json_path, md_path = report_dir / f"{REPORT_NAME}.json", report_dir / f"{REPORT_NAME}.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    md_path.write_text(render_markdown(summary))
    return json_path, md_path
