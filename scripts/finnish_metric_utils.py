"""Shared metric helpers for Finnish EQTransformer comparisons."""

import csv
import math
from pathlib import Path

import numpy as np


CSV_METRICS = [
    ("Matched event recall", "%"),
    ("P coverage", "%"),
    ("P MAE", "s"),
    ("S coverage (CSV first match)", "%"),
    ("S MAE (CSV first match)", "s"),
]

RAW_S_METRICS = [
    ("S coverage (all picker matches)", "%"),
    ("S MAE (all picker matches)", "s"),
    ("Raw S peak within 0.2 s", "%"),
    ("Raw S peak within 0.5 s", "%"),
    ("Raw S peak within 1.0 s", "%"),
    ("Raw S nearest-peak MAE", "s"),
]


def read_rows(path):
    path = Path(path)
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


def csv_metrics(rows, total):
    detected = {
        row[4]
        for row in rows
        if len(row) > 14 and row[14] not in ("", "0", "None")
    }
    result = {
        "Matched event recall": 100 * len(detected) / total,
    }

    p_errors = phase_errors(rows, 20)
    result["P coverage"] = 100 * len(p_errors) / total
    result["P MAE"] = (
        float(np.mean(np.abs(p_errors)) / 100) if len(p_errors) else math.nan
    )

    s_errors = phase_errors(rows, 24)
    result["S coverage (CSV first match)"] = 100 * len(s_errors) / total
    result["S MAE (CSV first match)"] = (
        float(np.mean(np.abs(s_errors)) / 100) if len(s_errors) else math.nan
    )
    return result


def raw_s_metrics(path, total):
    path = Path(path)
    if not path.exists():
        return {}

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    all_picker_errors = [
        abs(float(row["normal_picker_s_error_samples"])) / 100
        for row in rows
        if row.get("normal_picker_s_error_samples")
    ]
    raw_errors = [
        float(row["nearest_raw_s_abs_error_s"])
        for row in rows
        if row.get("nearest_raw_s_abs_error_s")
    ]

    result = {
        "S coverage (all picker matches)": 100 * len(all_picker_errors) / total,
        "S MAE (all picker matches)": (
            float(np.mean(all_picker_errors)) if all_picker_errors else math.nan
        ),
        "Raw S peak within 0.2 s": (
            100 * sum(error <= 0.2 for error in raw_errors) / total
        ),
        "Raw S peak within 0.5 s": (
            100 * sum(error <= 0.5 for error in raw_errors) / total
        ),
        "Raw S peak within 1.0 s": (
            100 * sum(error <= 1.0 for error in raw_errors) / total
        ),
        "Raw S nearest-peak MAE": (
            float(np.mean(raw_errors)) if raw_errors else math.nan
        ),
    }
    return result


def fmt(value, unit):
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if unit == "s":
        return f"{value:.4f} s"
    return f"{value:.2f}%"


def merge_metrics(csv_result, raw_result):
    result = dict(csv_result)
    result.update(raw_result)
    return result
