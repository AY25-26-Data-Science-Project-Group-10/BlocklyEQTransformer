#!/usr/bin/env python3
"""Diagnose raw S-picker peaks without requiring detector-window matching."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


CUSTOM_OBJECTS = {}


def load_runtime_dependencies():
    from keras.models import load_model

    from BlocklyEQTransformer.core.EqT_utils import (
        FeedForward,
        LayerNormalization,
        SeqSelfAttention,
        _detect_peaks,
        f1,
        picker,
    )

    custom_objects = {
        "SeqSelfAttention": SeqSelfAttention,
        "FeedForward": FeedForward,
        "LayerNormalization": LayerNormalization,
        "f1": f1,
    }
    return load_model, _detect_peaks, picker, custom_objects


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a model on a test split and inspect raw S probability peaks "
            "independently of detector-window matching."
        )
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--testset", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Optional directory for per-trace S probability plots.",
    )
    parser.add_argument(
        "--max-plots",
        default=0,
        type=int,
        help="Maximum number of plots to write. Default: 0.",
    )
    parser.add_argument(
        "--plot-window-samples",
        default=700,
        type=int,
        help="Half-window around true S for plots. Default: 700 samples.",
    )
    parser.add_argument(
        "--s-threshold",
        default=0.1,
        type=float,
        help="Minimum raw S probability peak height. Default: 0.1.",
    )
    parser.add_argument(
        "--p-threshold",
        default=0.1,
        type=float,
        help="P threshold passed to normal picker matching. Default: 0.1.",
    )
    parser.add_argument(
        "--detection-threshold",
        default=0.2,
        type=float,
        help="Detection threshold passed to normal picker matching. Default: 0.2.",
    )
    parser.add_argument(
        "--near-samples",
        default=50,
        type=int,
        help="Count raw S peak as near true S if abs error is <= this. Default: 50.",
    )
    parser.add_argument(
        "--input-dim",
        default=6000,
        type=int,
        help="Input sample count. Default: 6000.",
    )
    parser.add_argument(
        "--channels",
        default=3,
        type=int,
        help="Input channel count. Default: 3.",
    )
    parser.add_argument(
        "--normalization-mode",
        default="std",
        choices=["std", "max"],
        help="Normalization mode used by validation. Default: std.",
    )
    parser.add_argument(
        "--phase-types",
        default="d,P,S",
        help="Comma-separated model output order. Default: d,P,S.",
    )
    return parser.parse_args()


def station(trace_name):
    return str(trace_name).split("_")[0].split(".")[0]


def probability_outputs(model, hdf5_path, trace_names, args):
    from BlocklyEQTransformer.core.EqT_utils import DataGeneratorTest

    generator = DataGeneratorTest(
        trace_names,
        file_name=str(hdf5_path),
        dim=args.input_dim,
        batch_size=len(trace_names),
        n_channels=args.channels,
        norm_mode=args.normalization_mode,
    )
    prediction = model.predict_generator(generator=generator)
    if not isinstance(prediction, list):
        prediction = [prediction]

    phase_types = [item.strip() for item in args.phase_types.split(",") if item.strip()]
    output_by_phase = {}
    for index, phase in enumerate(phase_types):
        if index >= len(prediction):
            break
        output_by_phase[phase] = prediction[index].reshape(
            prediction[index].shape[0], prediction[index].shape[1]
        )

    missing = [phase for phase in ("d", "P", "S") if phase not in output_by_phase]
    if missing:
        raise ValueError(
            "Missing model outputs for phases: {}. Use --phase-types to set "
            "the model output order.".format(",".join(missing))
        )
    return output_by_phase["d"], output_by_phase["P"], output_by_phase["S"]


def nearest_peak(peaks, probabilities, true_sample):
    if len(peaks) == 0 or true_sample is None:
        return None, None, None
    nearest = min(peaks, key=lambda peak: abs(int(peak) - int(true_sample)))
    error = int(true_sample) - int(nearest)
    return int(nearest), float(probabilities[int(nearest)]), int(error)


def normal_picker_s_result(args, picker_func, d_prob, p_prob, s_prob, true_p, true_s):
    picker_args = {
        "detection_threshold": args.detection_threshold,
        "P_threshold": args.p_threshold,
        "S_threshold": args.s_threshold,
        "estimate_uncertainty": False,
    }
    zeros = np.zeros_like(d_prob)
    matches, pick_errors, _ = picker_func(
        picker_args,
        d_prob,
        p_prob,
        s_prob,
        zeros,
        zeros,
        zeros,
        true_p,
        true_s,
    )
    for key in matches:
        s_pick = matches[key][6]
        s_error = pick_errors.get(key, [None, None])[1]
        if s_pick is not None or s_error is not None:
            return int(s_pick) if s_pick is not None else None, s_error
    return None, None


def plot_trace(path, trace_name, true_s, s_prob, raw_peaks, nearest_raw, accepted_s, args):
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    start = max(0, int(true_s) - args.plot_window_samples)
    end = min(len(s_prob), int(true_s) + args.plot_window_samples)
    x = np.arange(start, end)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, s_prob[start:end], color="tab:blue", linewidth=1.1, label="raw S prob")
    ax.axhline(args.s_threshold, color="0.5", linestyle="--", linewidth=0.9, label="S threshold")
    ax.axvline(true_s, color="black", linewidth=1.0, label="true S")
    if nearest_raw is not None:
        ax.axvline(nearest_raw, color="tab:orange", linewidth=1.0, label="nearest raw S peak")
    if accepted_s is not None:
        ax.axvline(accepted_s, color="tab:green", linewidth=1.0, label="normal picker S")

    visible_peaks = [peak for peak in raw_peaks if start <= peak < end]
    if visible_peaks:
        ax.scatter(
            visible_peaks,
            s_prob[visible_peaks],
            color="tab:red",
            s=18,
            zorder=3,
            label="raw S peaks",
        )

    ax.set_title(trace_name)
    ax.set_xlabel("sample")
    ax.set_ylabel("S probability")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def main():
    args = parse_args()
    load_model, detect_peaks, picker_func, custom_objects = load_runtime_dependencies()
    import h5py

    trace_names = [str(item) for item in np.load(args.testset, allow_pickle=True)]
    model = load_model(str(args.model), custom_objects=custom_objects)
    d_prob, p_prob, s_prob = probability_outputs(model, args.hdf5, trace_names, args)

    rows = []
    with h5py.File(args.hdf5, "r") as h5:
        for index, trace_name in enumerate(trace_names):
            attrs = h5["data"][trace_name].attrs
            true_p = int(attrs["p_arrival_sample"]) if "p_arrival_sample" in attrs else None
            true_s = int(attrs["s_arrival_sample"]) if "s_arrival_sample" in attrs else None

            raw_peaks = detect_peaks(s_prob[index], mph=args.s_threshold, mpd=1)
            nearest_raw, nearest_raw_prob, nearest_raw_error = nearest_peak(
                raw_peaks, s_prob[index], true_s
            )
            accepted_s, accepted_s_error = normal_picker_s_result(
                args,
                picker_func,
                d_prob[index],
                p_prob[index],
                s_prob[index],
                true_p,
                true_s,
            )
            max_s_index = int(np.argmax(s_prob[index]))
            max_s_prob = float(s_prob[index][max_s_index])

            rows.append(
                {
                    "trace_name": trace_name,
                    "station": station(trace_name),
                    "true_s_sample": true_s,
                    "raw_s_peak_count": len(raw_peaks),
                    "max_s_sample": max_s_index,
                    "max_s_probability": max_s_prob,
                    "nearest_raw_s_sample": nearest_raw,
                    "nearest_raw_s_probability": nearest_raw_prob,
                    "nearest_raw_s_error_samples": nearest_raw_error,
                    "nearest_raw_s_abs_error_s": (
                        abs(nearest_raw_error) / 100 if nearest_raw_error is not None else None
                    ),
                    "nearest_raw_s_within_near_samples": (
                        abs(nearest_raw_error) <= args.near_samples
                        if nearest_raw_error is not None
                        else False
                    ),
                    "normal_picker_s_sample": accepted_s,
                    "normal_picker_s_error_samples": accepted_s_error,
                    "normal_picker_s_abs_error_s": (
                        abs(float(accepted_s_error)) / 100
                        if accepted_s_error is not None
                        else None
                    ),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})

    if args.plot_dir and args.max_plots > 0:
        ranked = sorted(
            rows,
            key=lambda row: (
                row["nearest_raw_s_abs_error_s"]
                if row["nearest_raw_s_abs_error_s"] is not None
                else float("inf"),
                -row["max_s_probability"],
            ),
            reverse=True,
        )
        row_by_trace = {row["trace_name"]: row for row in rows}
        for row in ranked[: args.max_plots]:
            trace_name = row["trace_name"]
            index = trace_names.index(trace_name)
            raw_peaks = detect_peaks(s_prob[index], mph=args.s_threshold, mpd=1)
            safe_name = trace_name.replace("/", "_")
            plot_trace(
                args.plot_dir / "{}.png".format(safe_name),
                trace_name,
                int(row_by_trace[trace_name]["true_s_sample"]),
                s_prob[index],
                raw_peaks,
                row_by_trace[trace_name]["nearest_raw_s_sample"],
                row_by_trace[trace_name]["normal_picker_s_sample"],
                args,
            )

    raw_with_peak = sum(1 for row in rows if row["raw_s_peak_count"] > 0)
    raw_near = sum(1 for row in rows if row["nearest_raw_s_within_near_samples"])
    normal_accepted = sum(1 for row in rows if row["normal_picker_s_error_samples"] is not None)
    print("Traces: {}".format(len(rows)))
    print(
        "Raw S peak >= threshold: {} ({:.2f}%)".format(
            raw_with_peak, 100 * raw_with_peak / len(rows)
        )
    )
    print(
        "Nearest raw S peak within {} samples: {} ({:.2f}%)".format(
            args.near_samples, raw_near, 100 * raw_near / len(rows)
        )
    )
    print(
        "S accepted by normal picker: {} ({:.2f}%)".format(
            normal_accepted, 100 * normal_accepted / len(rows)
        )
    )
    print("Wrote {}".format(args.output_csv))


if __name__ == "__main__":
    main()
