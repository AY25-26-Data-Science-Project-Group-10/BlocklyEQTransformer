#!/usr/bin/env python3
"""Print Finnish explosion EQT fine-tuning evaluation metrics."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return list(reader)


def phase_errors(rows, column):
    errors = []
    for row in rows:
        if len(row) > column and row[column] not in ("", "None"):
            errors.append(float(row[column]))
    return np.array(errors, dtype=float)


def metrics(rows, total):
    detected = {
        row[4]
        for row in rows
        if len(row) > 14 and row[14] not in ("", "0", "None")
    }
    result = {
        "Matched event recall": 100 * len(detected) / total,
    }

    # Column indices match this repository's tester output CSV layout.
    for phase, column in [("P", 20), ("S", 24)]:
        errors = phase_errors(rows, column)
        result[f"{phase} coverage"] = 100 * len(errors) / total
        result[f"{phase} MAE"] = (
            float(np.mean(np.abs(errors)) / 100) if len(errors) else math.nan
        )

    return result


def fmt(value, unit):
    if math.isnan(value):
        return "n/a"
    if unit == "s":
        return f"{value:.4f} s"
    return f"{value:.2f}%"


def load_metrics(label, path, total):
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Cannot compute {label} metrics: no result rows in {path}")
    return metrics(rows, total)


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
        default=None,
        type=Path,
        help=(
            "Optional pretrained-baseline X_test_results.csv. If supplied, "
            "the output table compares pretrained and fine-tuned metrics."
        ),
    )
    args = parser.parse_args()

    total = len(np.load(args.testset, allow_pickle=True))

    try:
        finetuned = load_metrics("fine-tuned", args.finetuned, total)
        pretrained = (
            load_metrics("pretrained", args.pretrained, total)
            if args.pretrained is not None
            else None
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    rows = [
        ("Matched event recall", "%"),
        ("P coverage", "%"),
        ("P MAE", "s"),
        ("S coverage", "%"),
        ("S MAE", "s"),
    ]

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
