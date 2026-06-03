#!/usr/bin/env python3
"""Print pretrained vs fine-tuned Finnish EQT comparison metrics."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from finnish_metric_utils import (
    CSV_METRICS,
    RAW_S_METRICS,
    csv_metrics,
    fmt,
    merge_metrics,
    raw_s_metrics,
    read_rows,
)


def report_value(results_path, key):
    path = results_path.with_name("X_report.txt")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        prefix = f"{key}: "
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def resolve_report_path(value, notebook_relative=True):
    if value is None:
        return None
    path = Path(value)
    if notebook_relative and not path.is_absolute():
        path = Path("notebook") / path
    return path


def ensure_raw_s(args, label, model, output_csv):
    if not args.compute_raw_s:
        return
    if model is None:
        raise ValueError(f"Cannot compute {label} raw-S metrics without a model path")
    if output_csv.exists() and not args.overwrite_raw_s:
        return
    command = [
        sys.executable,
        "scripts/diagnose_raw_s_picks.py",
        "--model",
        str(model),
        "--hdf5",
        str(args.hdf5),
        "--testset",
        str(args.testset),
        "--output-csv",
        str(output_csv),
        "--s-threshold",
        str(args.s_threshold),
        "--p-threshold",
        str(args.p_threshold),
        "--detection-threshold",
        str(args.detection_threshold),
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Print comparison metrics for Finnish EQT evaluations."
    )
    parser.add_argument(
        "--testset",
        default="notebook/eq_finetune_outputs/test.npy",
        type=Path,
    )
    parser.add_argument(
        "--pretrained",
        default="notebook/finnish_pretrained_eval_outputs/X_test_results.csv",
        type=Path,
    )
    parser.add_argument(
        "--finetuned",
        default="notebook/finnish_finetuned_best_eval_outputs/X_test_results.csv",
        type=Path,
    )
    parser.add_argument(
        "--hdf5",
        default="data/finnish_eq_finetune.hdf5",
        type=Path,
    )
    parser.add_argument(
        "--pretrained-model",
        default=None,
        type=Path,
        help="Model path for --compute-raw-s. Defaults to value in pretrained X_report.txt.",
    )
    parser.add_argument(
        "--finetuned-model",
        default=None,
        type=Path,
        help="Model path for --compute-raw-s. Defaults to value in fine-tuned X_report.txt.",
    )
    parser.add_argument(
        "--pretrained-raw-s",
        default="notebook/eq_pretrained_raw_s_diagnostic.csv",
        type=Path,
    )
    parser.add_argument(
        "--finetuned-raw-s",
        default="notebook/eq_raw_s_diagnostic.csv",
        type=Path,
    )
    parser.add_argument("--compute-raw-s", action="store_true")
    parser.add_argument("--overwrite-raw-s", action="store_true")
    parser.add_argument("--detection-threshold", default=0.2, type=float)
    parser.add_argument("--p-threshold", default=0.1, type=float)
    parser.add_argument("--s-threshold", default=0.1, type=float)
    args = parser.parse_args()

    total = len(np.load(args.testset, allow_pickle=True))
    try:
        pretrained_rows = read_rows(args.pretrained)
        finetuned_rows = read_rows(args.finetuned)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    if not pretrained_rows or not finetuned_rows:
        sys.exit("Both pretrained and fine-tuned results must contain rows.")

    pretrained_model = args.pretrained_model or resolve_report_path(
        report_value(args.pretrained, "input_model"), notebook_relative=True
    )
    finetuned_model = args.finetuned_model or resolve_report_path(
        report_value(args.finetuned, "input_model"), notebook_relative=True
    )

    try:
        ensure_raw_s(args, "pretrained", pretrained_model, args.pretrained_raw_s)
        ensure_raw_s(args, "fine-tuned", finetuned_model, args.finetuned_raw_s)
    except (subprocess.CalledProcessError, ValueError) as exc:
        sys.exit(str(exc))

    pretrained = merge_metrics(
        csv_metrics(pretrained_rows, total),
        raw_s_metrics(args.pretrained_raw_s, total),
    )
    finetuned = merge_metrics(
        csv_metrics(finetuned_rows, total),
        raw_s_metrics(args.finetuned_raw_s, total),
    )

    rows = CSV_METRICS + RAW_S_METRICS

    print(f"Test traces: {total}")
    print()
    print("| Metric | Pretrained | Fine-tuned best |")
    print("| --- | ---: | ---: |")
    for name, unit in rows:
        print(f"| {name} | {fmt(pretrained[name], unit)} | {fmt(finetuned[name], unit)} |")


if __name__ == "__main__":
    main()
