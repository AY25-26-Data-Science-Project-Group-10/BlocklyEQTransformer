#!/usr/bin/env python3
"""Print Finnish explosion EQT fine-tuning evaluation metrics."""

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


def load_metrics(label, path, total):
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Cannot compute {label} metrics: no result rows in {path}")
    return csv_metrics(rows, total)


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
        description="Print metrics for Finnish explosion EQT evaluations."
    )
    parser.add_argument(
        "--testset",
        default="notebook/expl_finetune_outputs/test.npy",
        type=Path,
    )
    parser.add_argument(
        "--finetuned",
        default="notebook/finnish_expl_finetuned_best_eval_outputs/X_test_results.csv",
        type=Path,
    )
    parser.add_argument(
        "--pretrained",
        default="notebook/finnish_expl_pretrained_eval_outputs/X_test_results.csv",
        type=Path,
        help=(
            "Pretrained-baseline X_test_results.csv. Use an empty string to "
            "print only fine-tuned metrics."
        ),
    )
    parser.add_argument(
        "--hdf5",
        default="data/finnish_explosion_finetune.hdf5",
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
        default="notebook/expl_pretrained_raw_s_diagnostic.csv",
        type=Path,
    )
    parser.add_argument(
        "--finetuned-raw-s",
        default="notebook/expl_raw_s_diagnostic.csv",
        type=Path,
    )
    parser.add_argument("--compute-raw-s", action="store_true")
    parser.add_argument("--overwrite-raw-s", action="store_true")
    parser.add_argument("--detection-threshold", default=0.2, type=float)
    parser.add_argument("--p-threshold", default=0.1, type=float)
    parser.add_argument("--s-threshold", default=0.1, type=float)
    args = parser.parse_args()

    total = len(np.load(args.testset, allow_pickle=True))
    if str(args.pretrained) == "":
        args.pretrained = None

    try:
        finetuned_rows = read_rows(args.finetuned)
        pretrained_rows = read_rows(args.pretrained) if args.pretrained else None
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    finetuned_model = args.finetuned_model or resolve_report_path(
        report_value(args.finetuned, "input_model"), notebook_relative=True
    )
    pretrained_model = (
        args.pretrained_model
        or (
            resolve_report_path(report_value(args.pretrained, "input_model"), notebook_relative=True)
            if args.pretrained
            else None
        )
    )

    try:
        ensure_raw_s(args, "fine-tuned", finetuned_model, args.finetuned_raw_s)
        if args.pretrained:
            ensure_raw_s(args, "pretrained", pretrained_model, args.pretrained_raw_s)
    except (subprocess.CalledProcessError, ValueError) as exc:
        sys.exit(str(exc))

    finetuned = merge_metrics(
        csv_metrics(finetuned_rows, total),
        raw_s_metrics(args.finetuned_raw_s, total),
    )
    pretrained = (
        merge_metrics(
            csv_metrics(pretrained_rows, total),
            raw_s_metrics(args.pretrained_raw_s, total),
        )
        if pretrained_rows is not None
        else None
    )

    rows = CSV_METRICS + RAW_S_METRICS

    print(f"Test traces: {total}")
    print()
    if pretrained is None:
        print("| Metric | Fine-tuned best |")
        print("| --- | ---: |")
        for name, unit in rows:
            print(f"| {name} | {fmt(finetuned[name], unit)} |")
    else:
        print("| Metric | Pretrained | Fine-tuned best |")
        print("| --- | ---: | ---: |")
        for name, unit in rows:
            print(
                f"| {name} | {fmt(pretrained[name], unit)} | "
                f"{fmt(finetuned[name], unit)} |"
            )


if __name__ == "__main__":
    main()
