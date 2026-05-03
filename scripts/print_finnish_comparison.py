#!/usr/bin/env python3
"""Print pretrained vs fine-tuned Finnish EQT comparison metrics."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def report_path(results_path):
    return results_path.with_name("X_report.txt")


def report_summary(results_path):
    path = report_path(results_path)
    if not path.exists():
        return []

    interesting = (
        "input_model:",
        "input_testset:",
        "detection_threshold:",
        "P_threshold:",
        "S_threshold:",
        "number_of_plots:",
    )
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith(interesting):
            lines.append(line)
    return lines


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
    args = parser.parse_args()

    total = len(np.load(args.testset, allow_pickle=True))
    try:
        pretrained_rows = read_rows(args.pretrained)
        finetuned_rows = read_rows(args.finetuned)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    missing = []
    if not pretrained_rows:
        missing.append(("pretrained", args.pretrained))
    if not finetuned_rows:
        missing.append(("fine-tuned", args.finetuned))

    if missing:
        print(f"Test traces: {total}")
        print()
        for label, path in missing:
            print(f"Cannot compute {label} metrics: no result rows in {path}")
            details = report_summary(path)
            if details:
                print("Validation report:")
                for line in details:
                    print(f"  {line}")
            print()
        print(
            "Rerun validation for the intended checkpoint, or lower thresholds, "
            "until X_test_results.csv contains detected/picked rows."
        )
        sys.exit(1)

    pretrained = metrics(pretrained_rows, total)
    finetuned = metrics(finetuned_rows, total)

    rows = [
        ("Matched event recall", "%"),
        ("P coverage", "%"),
        ("P MAE", "s"),
        ("S coverage", "%"),
        ("S MAE", "s"),
    ]

    print(f"Test traces: {total}")
    print()
    print("| Metric | Pretrained | Fine-tuned best |")
    print("| --- | ---: | ---: |")
    for name, unit in rows:
        print(f"| {name} | {fmt(pretrained[name], unit)} | {fmt(finetuned[name], unit)} |")


if __name__ == "__main__":
    main()
