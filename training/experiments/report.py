"""Tracked summary of the three-dataset retention experiment (JSON + markdown).

Every number comes from the per-dataset ``combined.json``; nothing is recomputed or typed by hand.
Failed or missing pieces are rendered as "—".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "evaluation" / "reports" / "experiments"
REPORT_NAME = "retention_datasets"
VARIANT_ORDER = ("majority", "oracle", "xgb_asis", "xgb_fitonly", "deep_asis", "shallow_asis", "deep_seedavg",
                 "shallow_seedavg", "ensemble", "thr_xgb_fitonly", "thr_deep_asis", "thr_shallow_asis",
                 "thr_ensemble", "xgb_tertile")
# shipped MSC numbers the reproduction is checked against (plan "Existing code facts")
REFERENCE = {"msc": {"xgb_asis": 0.611397, "deep_asis": 0.673240, "shallow_asis": 0.650437}}
CAVEATS = (
    "CSVs are template/AI-generated with synthetic scores",
    "label-noise ceilings (oracle)",
    "bin edges are a modelling choice",
    "effective test sizes ({sizes})",
    "validation reuse / multiple comparisons",
    "MSC rows are a reproduction, not a new result",
    "nothing in `main/`, `artifacts/importance*` or the shipped reports changed",
)
ORACLE_WORDING = ("The oracle uses the test labels, is the best any deterministic text function can score, "
                  "is not a model, and a model exceeding it indicates a bug.")


def num(value, digits=4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def ci(interval, digits=3) -> str:
    return "—" if not interval else f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def tuned_text(record: dict) -> str:
    tuned = record.get("tuned") or {}
    parts = []
    for key in ("max_depth", "min_child_weight", "best_iteration", "seed", "best_epoch", "w"):
        if key in tuned:
            parts.append(f"{key}={tuned[key]}")
    for key in ("temperature", "ensemble_temperature", "xgb_temperature"):
        if key in tuned and tuned[key] is not None:
            parts.append(f"{'T' if key == 'temperature' else key}={tuned[key]:.3f}")
    if "seeds" in tuned:
        parts.append("seeds=" + "/".join(str(s) for s in tuned["seeds"]))
    if "class_weights" in tuned:
        parts.append("cw=" + "/".join(f"{w:g}" for w in tuned["class_weights"]))
    if "class_name" in tuned:
        parts.append(tuned["class_name"])
    if "bin_edges" in tuned and tuned["bin_edges"]:
        parts.append("edges=" + "/".join(f"{e:.2f}" for e in tuned["bin_edges"]))
    return ", ".join(parts) or "—"


def build_summary(dataset_results: dict, manifest: dict | None = None) -> dict:
    """dataset_results: name -> combined.json dict, or None when the dataset's pipeline failed."""
    datasets = {}
    for name, combined in dataset_results.items():
        if not combined:
            datasets[name] = {"status": "failed"}
            continue
        reference = REFERENCE.get(name, {})
        datasets[name] = {
            "status": "ok", "kind": combined["kind"], "class_names": combined["class_names"],
            "stage0": {key: combined["stage0"].get(key) for key in
                       ("n_rows", "omitted_rows", "bin_edges", "tertile_edges", "normalisation", "pin_fraction",
                        "pinned_groups", "split_protocol", "split_sizes", "split_sentences", "class_counts", "seed")},
            "diagnostics": combined["stage0"]["diagnostics"],
            "resamples": combined["resamples"], "bootstrap_seed": combined["bootstrap_seed"],
            "variants": combined["variants"], "comparisons": combined["comparisons"],
            "blend_table": combined.get("blend_table"), "seeds": combined.get("seeds", {}),
            "xgboost_trials": combined.get("xgboost_trials", {}), "missing": combined.get("missing", []),
            "self_checks": combined.get("self_checks", {}),
            "reproduction": {variant: {"reference": expected,
                                       "observed": combined["variants"].get(variant, {}).get("test", {}).get("macro_f1")}
                             for variant, expected in reference.items()},
        }
        for item in datasets[name]["reproduction"].values():
            item["delta"] = None if item["observed"] is None else item["observed"] - item["reference"]
    sizes = []
    for name, item in datasets.items():
        if item["status"] == "ok" and item["kind"] == "csv":
            pinned = item["stage0"]["pinned_groups"] or []
            note = f"{name}: {item['stage0']['split_sentences']['test']} test sentences"
            if pinned:
                note += ", " + "; ".join(f"{p['rows']}-row sentence pinned to fit" for p in pinned)
            sizes.append(note)
    return {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "manifest": manifest,
            "datasets": datasets,
            "caveats": [c.format(sizes="; ".join(sizes) or "—") for c in CAVEATS], "oracle_wording": ORACLE_WORDING}


def _diagnostics_lines(name: str, item: dict) -> list[str]:
    stage0, diag = item["stage0"], item["diagnostics"]
    counts = stage0["class_counts"]
    pinned = stage0["pinned_groups"] or []
    lines = ["| Diagnostic | Value |", "| --- | --- |",
             f"| rows / unique sentences | {diag['rows']} / {diag['unique_sentences']} |",
             f"| singleton / multi-row groups | {diag['singleton_groups']} / {diag['multi_row_groups']} |",
             f"| largest group | {diag['largest_group']['rows']} rows |",
             f"| multi-row groups disagreeing on class | {diag['disagreeing_multi_groups']} ({num(diag['disagreeing_multi_group_share'], 3)}) |",
             f"| mean within-group score std | {num(diag['mean_within_group_score_std'])} |",
             f"| bin edges / tertile edges | {stage0['bin_edges'] or '—'} / {stage0['tertile_edges'] or '—'} |",
             f"| text normalisation | {stage0['normalisation']} |",
             f"| split protocol / seed | {stage0['split_protocol']} / {stage0['seed']} |",
             f"| pin rule | {stage0['pin_fraction']} → " + (", ".join(f"{p['rows']} rows" for p in pinned) or "nothing pinned") + " |",
             f"| fit / val / test rows | {stage0['split_sizes']['fit']} / {stage0['split_sizes']['val']} / {stage0['split_sizes']['test']} |",
             f"| fit / val / test sentences | {stage0['split_sentences']['fit']} / {stage0['split_sentences']['val']} / {stage0['split_sentences']['test']} |"]
    for split in ("all", "fit", "val", "test"):
        lines.append(f"| class counts ({split}) | " + ", ".join(f"{k} {v}" for k, v in counts[split].items()) + " |")
    lines.append(f"| majority class (fit+val) | {diag['majority']['class_name']} |")
    return lines


def _variants_table(item: dict) -> list[str]:
    lines = ["| Variant | Val macro-F1 | Test acc | Bal acc | Macro-F1 [row CI] [sentence CI] | NLL | ECE | Tuned |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    for name in VARIANT_ORDER:
        record = item["variants"].get(name)
        if not record or record.get("status") != "ok":
            lines.append(f"| {name} | — | — | — | — | — | — | — |")
            continue
        test, val = record["test"], record["val"]
        f1 = f"{num(test['macro_f1'])} {ci(record['row_bootstrap']['macro_f1_ci95'])} " \
             f"{ci(record['sentence_bootstrap']['macro_f1_ci95']) if record.get('sentence_bootstrap') else '—'}"
        label = name + (" (exploratory)" if record.get("exploratory") else "") + (" (tertile labels)" if record.get("labels") == "tertile" else "")
        lines.append(f"| {label} | {num(val['macro_f1'])} | {num(test['accuracy'])} | {num(test['balanced_accuracy'])} | {f1} | "
                     f"{num(test.get('nll'))} | {num(test.get('ece_10_bins'))} | {tuned_text(record)} |")
    return lines


def _comparisons_table(item: dict) -> list[str]:
    lines = ["| Baseline → candidate | Δ macro-F1 | row CI | sentence CI | one-sided p | Δ accuracy | acc CI | acc p | Primary |",
             "| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |"]
    for comp in item["comparisons"]:
        label = f"{comp['baseline']} → {comp['candidate']}"
        primary = f"yes (α={comp['alpha']})" if comp["primary"] else "exploratory"
        if comp.get("status") != "ok":
            lines.append(f"| {label} | — | — | — | — | — | — | — | {primary} |")
            continue
        f1, acc, sent = comp["macro_f1"], comp["accuracy"], comp.get("macro_f1_sentence")
        lines.append(f"| {label} | {num(f1['observed'])} | {ci(f1['ci95'])} | {ci(sent['ci95']) if sent else '—'} | "
                     f"{num(f1['p_one_sided'])} | {num(acc['mean_difference'])} | {ci(acc['ci95'])} | {num(acc['p_value'])} | {primary} |")
    return lines


def _seed_table(item: dict) -> list[str]:
    lines = ["| Head | Seed | Best epoch | Val macro-F1 | Val NLL | T |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for short in ("deep", "shallow"):
        seeds = item["seeds"].get(short) or []
        temps = (item["variants"].get(f"{short}_seedavg") or {}).get("tuned", {}).get("per_seed_temperature", {})
        if not seeds:
            lines.append(f"| {short} | — | — | — | — | — |")
        for info in seeds:
            lines.append(f"| {short} | {info.get('seed')} | {info.get('best_epoch', '—')} | {num(info.get('validation_macro_f1'))} | "
                         f"{num(info.get('validation_nll'))} | {num(temps.get(str(info.get('seed'))))} |")
    return lines


def _trials_table(item: dict) -> list[str]:
    lines = ["| Labels | max_depth | min_child_weight | Best iteration | Val macro-F1 | Val balanced acc |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in (("xgboost", "main"), ("xgboost_tertile", "tertile")):
        trials = item["xgboost_trials"].get(key)
        if not trials:
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        for trial in trials:
            lines.append(f"| {label} | {trial.get('max_depth')} | {trial.get('min_child_weight')} | {trial.get('best_iteration')} | "
                         f"{num(trial.get('macro_f1'))} | {num(trial.get('balanced_accuracy'))} |")
    return lines


def _sentence_table(item: dict) -> list[str]:
    lines = ["| Variant | Sentences | Accuracy | Balanced acc | Macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in VARIANT_ORDER:
        record = item["variants"].get(name) or {}
        level = record.get("sentence_level")
        if level:
            lines.append(f"| {name} | {sum(c['support'] for c in level['per_class'].values())} | {num(level['accuracy'])} | "
                         f"{num(level['balanced_accuracy'])} | {num(level['macro_f1'])} |")
    return lines


def render_markdown(summary: dict) -> str:
    lines = ["# Retention models on three datasets (isolated experiment)", "",
             f"Generated {summary['created_at']}. Shipped artifacts and reports are untouched; every number below is "
             "copied from `artifacts/experiments/retention_datasets/<dataset>/combined.json`.", ""]
    manifest = summary.get("manifest") or {}
    if manifest:
        failed = [f"{step['name']}" for step in manifest.get("steps", []) if step.get("exit_code") not in (0, None)]
        lines += [f"Run: git `{manifest.get('git_head', '—')}`, argv `{' '.join(manifest.get('argv', [])) or '—'}`, "
                  f"failed steps: {', '.join(failed) or 'none'}.", ""]
    for name, item in summary["datasets"].items():
        lines += [f"## {name}", ""]
        if item.get("status") != "ok":
            lines += ["Pipeline failed; no results. —", ""]
            continue
        lines += ["### Diagnostics", ""] + _diagnostics_lines(name, item) + ["", ORACLE_WORDING, ""]
        lines += ["### Variants (tuned on validation only; val macro-F1 is optimistic)", ""] + _variants_table(item) + [""]
        if item.get("blend_table"):
            lines += ["Ensemble weight search on validation:", "", "| w (XGBoost share) | Val macro-F1 | Val NLL |", "| ---: | ---: | ---: |"]
            lines += [f"| {row['w']} | {num(row['macro_f1'])} | {num(row['nll'])} |" for row in item["blend_table"]] + [""]
        lines += ["### Paired comparisons (candidate − baseline on the shared bootstrap matrix)", ""] + _comparisons_table(item) + [""]
        lines += ["### Neural seeds (validation)", ""] + _seed_table(item) + [""]
        lines += ["### XGBoost validation trials", ""] + _trials_table(item) + [""]
        if item["kind"] == "csv":
            lines += ["### Sentence-level secondary table (test deduplicated by sentence, label = test majority)", ""] + _sentence_table(item) + [""]
        replay = (manifest.get("replay") or {}) if item["kind"] == "msc" else {}
        if replay:
            lines += ["### Replay of the shipped artifacts through this harness (must match exactly)", "",
                      "| Artifact | Macro-F1 | Confusion matrix | Exact match |", "| --- | ---: | --- | --- |"]
            lines += [f"| {artifact} | {num(r['metrics']['macro_f1'])} | `{r['metrics']['confusion_matrix']}` | "
                      f"{'yes' if r['exact_match'] else 'NO'} |" for artifact, r in replay.items()] + [""]
        if item.get("reproduction"):
            lines += ["### Reproduction of the shipped MSC numbers (retrained here)", "", "| Variant | Shipped macro-F1 | This run | Δ |", "| --- | ---: | ---: | ---: |"]
            lines += [f"| {variant} | {num(r['reference'])} | {num(r['observed'])} | {num(r['delta'])} |"
                      for variant, r in item["reproduction"].items()] + [""]
        checks = item.get("self_checks") or {}
        lines += ["Self-checks: " + ", ".join(f"{k}={'—' if v is None else v}" for k, v in checks.items()) + ".",
                  "Missing pieces: " + (", ".join(item.get("missing") or []) or "none") + ".", ""]
    lines += ["## Caveats", ""] + [f"- {caveat}" for caveat in summary["caveats"]] + [""]
    return "\n".join(lines)


def write(summary: dict, report_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = report_dir / f"{REPORT_NAME}.json", report_dir / f"{REPORT_NAME}.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path.write_text(render_markdown(summary))
    return json_path, md_path
