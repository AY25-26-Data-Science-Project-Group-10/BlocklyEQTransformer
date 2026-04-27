#!/usr/bin/env python3
"""Convert EQCCTPro-style Finnish waveform folders to BlocklyEQTransformer HDF5/CSV.

The converter expects waveform folders like:

    waveforms_earthquakes/<START>_<END>/<STATION>/*.mseed

and label CSV rows whose ``file_name`` looks like:

    NET.STA.LOC.CHAN | 2025-01-01T09:22:02.766970Z - 2025-01-01T09:23:02.766970Z

It writes the STEAD-like layout used by BlocklyEQTransformer:

    HDF5: /data/{trace_name}
    CSV:  one row per trace with a trace_name column
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from obspy import UTCDateTime, read


LABEL_RE = re.compile(
    r"^(?P<network>[^.]+)\."
    r"(?P<station>[^.]+)\."
    r"(?P<location>[^.]*)\."
    r"(?P<channel>\S+)\s*\|\s*"
    r"(?P<start>\S+)\s*-\s*"
    r"(?P<end>\S+)\s*$"
)

CSV_COLUMNS = [
    "network_code",
    "receiver_code",
    "receiver_type",
    "receiver_latitude",
    "receiver_longitude",
    "receiver_elevation_m",
    "p_arrival_sample",
    "p_status",
    "p_weight",
    "p_travel_sec",
    "s_arrival_sample",
    "s_status",
    "s_weight",
    "source_id",
    "source_origin_time",
    "source_origin_uncertainty_sec",
    "source_latitude",
    "source_longitude",
    "source_error_sec",
    "source_gap_deg",
    "source_horizontal_uncertainty_km",
    "source_depth_km",
    "source_depth_uncertainty_km",
    "source_magnitude",
    "source_magnitude_type",
    "source_magnitude_author",
    "source_mechanism_strike_dip_rake",
    "source_distance_deg",
    "source_distance_km",
    "back_azimuth_deg",
    "snr_db",
    "coda_end_sample",
    "trace_start_time",
    "trace_category",
    "trace_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EQCCTPro waveform folders and manual P/S labels into "
            "BlocklyEQTransformer fine-tuning HDF5 and CSV files."
        )
    )
    parser.add_argument(
        "--waveform-dir",
        default="data/waveforms_earthquakes",
        type=Path,
        help="Directory containing EQCCTPro waveform windows.",
    )
    parser.add_argument(
        "--labels",
        default=[Path("data/eq_labels.csv")],
        nargs="+",
        type=Path,
        help="One or more EQCCTPro label CSV files.",
    )
    parser.add_argument(
        "--stations-json",
        default=Path("station_list.json"),
        type=Path,
        help="Optional station metadata JSON with network/channels/coords.",
    )
    parser.add_argument(
        "--output-hdf5",
        default=Path("data/finnish_eq_finetune.hdf5"),
        type=Path,
        help="Output HDF5 file.",
    )
    parser.add_argument(
        "--output-csv",
        default=Path("data/finnish_eq_finetune.csv"),
        type=Path,
        help="Output CSV file.",
    )
    parser.add_argument(
        "--target-sampling-rate",
        default=100.0,
        type=float,
        help="Sampling rate expected by BlocklyEQTransformer.",
    )
    parser.add_argument(
        "--target-samples",
        default=6000,
        type=int,
        help="Number of samples per output trace.",
    )
    parser.add_argument(
        "--max-records",
        default=None,
        type=int,
        help="Convert at most this many matched station-window records.",
    )
    parser.add_argument(
        "--pick-policy",
        default="both",
        choices=["both", "any"],
        help=(
            "Use 'both' to keep only traces with valid P and S picks. "
            "Use 'any' to also keep traces with only one valid pick."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the generated HDF5 and CSV after conversion.",
    )
    return parser.parse_args()


def missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def parse_utc(value: object) -> Optional[UTCDateTime]:
    if missing(value):
        return None
    return UTCDateTime(str(value).strip())


def compact_utc(value: UTCDateTime) -> str:
    return value.datetime.strftime("%Y%m%dT%H%M%SZ")


def blockly_time(value: UTCDateTime) -> str:
    return value.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")


def label_key(station: str, start: UTCDateTime, end: UTCDateTime) -> Tuple[str, str]:
    return station, "{}_{}".format(compact_utc(start), compact_utc(end))


def parse_label_file_name(file_name: str) -> Optional[dict]:
    match = LABEL_RE.match(str(file_name).strip())
    if not match:
        return None
    parsed = match.groupdict()
    parsed["start_time"] = UTCDateTime(parsed["start"])
    parsed["end_time"] = UTCDateTime(parsed["end"])
    parsed["key"] = label_key(parsed["station"], parsed["start_time"], parsed["end_time"])
    return parsed


def add_pick(record: dict, phase: str, pick_time: UTCDateTime, row: dict, source: Path) -> None:
    current = record.get(phase)
    if current is None:
        record[phase] = pick_time
        record["pick_sources"][phase] = "{}:{}".format(source, row.get("_rownum", "?"))
        return
    if abs(current - pick_time) > 0.005:
        record["conflicts"].append(
            "{} pick conflict for {}: kept {}, ignored {} from {}:{}".format(
                phase.upper(),
                record["key"],
                current.isoformat(),
                pick_time.isoformat(),
                source,
                row.get("_rownum", "?"),
            )
        )


def load_labels(label_paths: Sequence[Path]) -> Tuple[Dict[Tuple[str, str], dict], List[str], int]:
    labels: Dict[Tuple[str, str], dict] = {}
    warnings: List[str] = []
    rows_seen = 0

    for label_path in label_paths:
        with label_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for rownum, row in enumerate(reader, start=2):
                rows_seen += 1
                row["_rownum"] = rownum
                parsed = parse_label_file_name(row.get("file_name", ""))
                if parsed is None:
                    warnings.append(
                        "Could not parse file_name at {}:{}: {}".format(
                            label_path, rownum, row.get("file_name", "")
                        )
                    )
                    continue

                key = parsed["key"]
                if key not in labels:
                    labels[key] = {
                        "key": key,
                        "network": parsed["network"],
                        "station": parsed["station"],
                        "location": parsed["location"],
                        "label_channel": parsed["channel"],
                        "label_start_time": parsed["start_time"],
                        "label_end_time": parsed["end_time"],
                        "p": None,
                        "s": None,
                        "pick_sources": {},
                        "conflicts": [],
                        "rows": 0,
                    }

                record = labels[key]
                record["rows"] += 1

                p_time = parse_utc(row.get("p_arrival_time"))
                if p_time is not None:
                    add_pick(record, "p", p_time, row, label_path)

                s_time = parse_utc(row.get("s_arrival_time"))
                if s_time is not None:
                    add_pick(record, "s", s_time, row, label_path)

    for record in labels.values():
        warnings.extend(record["conflicts"])

    return labels, warnings, rows_seen


def load_station_metadata(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def index_waveform_groups(waveform_dir: Path) -> Dict[Tuple[str, str], Path]:
    groups: Dict[Tuple[str, str], Path] = {}
    for station_dir in sorted(waveform_dir.glob("*/*")):
        if not station_dir.is_dir():
            continue
        key = (station_dir.name, station_dir.parent.name)
        groups[key] = station_dir
    return groups


def component_column(channel: str) -> Optional[int]:
    suffix = channel[-1].upper()
    if suffix in {"E", "1", "0"}:
        return 0
    if suffix in {"N", "2"}:
        return 1
    if suffix == "Z":
        return 2
    return None


def read_waveform_group(
    station_dir: Path,
    target_rate: float,
    target_samples: int,
) -> Tuple[np.ndarray, dict]:
    traces = []
    for mseed_path in sorted(station_dir.glob("*.mseed")):
        stream = read(str(mseed_path))
        for trace in stream:
            col = component_column(trace.stats.channel)
            if col is not None:
                traces.append((col, trace.copy(), mseed_path))

    by_component = {}
    for col, trace, path in traces:
        if col not in by_component:
            by_component[col] = (trace, path)

    missing_components = sorted(set([0, 1, 2]) - set(by_component))
    if missing_components:
        raise ValueError(
            "missing components {} in {}".format(missing_components, station_dir)
        )

    common_start = max(trace.stats.starttime for trace, _ in by_component.values())
    target_end = common_start + (target_samples - 1) / target_rate
    data = np.zeros((target_samples, 3), dtype=np.float32)

    for col, (trace, _path) in by_component.items():
        if abs(float(trace.stats.sampling_rate) - target_rate) > 1e-6:
            trace.interpolate(
                sampling_rate=target_rate,
                method="linear",
                starttime=trace.stats.starttime,
            )
        trace.trim(
            starttime=common_start,
            endtime=target_end,
            pad=True,
            fill_value=0,
            nearest_sample=True,
        )
        component_data = np.asarray(trace.data, dtype=np.float32)
        if component_data.size < target_samples:
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[: component_data.size] = component_data
            component_data = padded
        elif component_data.size > target_samples:
            component_data = component_data[:target_samples]
        data[:, col] = component_data

    z_trace = by_component[2][0]
    metadata = {
        "network": z_trace.stats.network or "",
        "station": z_trace.stats.station or station_dir.name,
        "location": z_trace.stats.location or "",
        "receiver_type": z_trace.stats.channel[:2] or "",
        "trace_start_time": common_start,
        "original_sampling_rates": [
            float(trace.stats.sampling_rate) for trace, _ in by_component.values()
        ],
    }
    return data, metadata


def sample_index(pick_time: Optional[UTCDateTime], start_time: UTCDateTime, rate: float, samples: int) -> Optional[int]:
    if pick_time is None:
        return None
    index = int(round((pick_time - start_time) * rate))
    if index <= 0 or index >= samples:
        return None
    return index


def finite_or_blank(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def station_coords(stations: dict, station: str) -> Tuple[float, float, float]:
    info = stations.get(station, {})
    coords = info.get("coords") or [None, None, None]
    values = []
    for value in coords[:3]:
        values.append(float(value) if value is not None else float("nan"))
    while len(values) < 3:
        values.append(float("nan"))
    return values[0], values[1], values[2]


def estimate_snr_db(data: np.ndarray, p_sample: Optional[int], s_sample: Optional[int]) -> np.ndarray:
    if p_sample is None:
        return np.zeros(3, dtype=np.float32)

    noise_end = max(1, p_sample - 50)
    noise_start = max(0, noise_end - 500)
    signal_start = p_sample
    if s_sample is not None:
        signal_end = min(data.shape[0], s_sample + 300)
    else:
        signal_end = min(data.shape[0], p_sample + 1000)

    if noise_end <= noise_start or signal_end <= signal_start:
        return np.zeros(3, dtype=np.float32)

    noise = data[noise_start:noise_end, :]
    signal = data[signal_start:signal_end, :]
    noise_rms = np.sqrt(np.mean(np.square(noise), axis=0))
    signal_rms = np.sqrt(np.mean(np.square(signal), axis=0))
    noise_rms[noise_rms == 0] = 1.0
    snr = 20.0 * np.log10(signal_rms / noise_rms)
    snr[~np.isfinite(snr)] = 0.0
    return snr.astype(np.float32)


def coda_end_sample(p_sample: Optional[int], s_sample: Optional[int], samples: int) -> int:
    if p_sample is not None and s_sample is not None and s_sample > p_sample:
        return min(samples - 1, int(round(s_sample + max(100, 1.4 * (s_sample - p_sample)))))
    if s_sample is not None:
        return min(samples - 1, s_sample + 1000)
    if p_sample is not None:
        return min(samples - 1, p_sample + 1000)
    return samples - 1


def snr_to_text(snr: np.ndarray) -> str:
    return "[{}]".format(" ".join("{:.4f}".format(float(value)) for value in snr))


def h5_attr_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def write_outputs(
    matched_items: Sequence[Tuple[Tuple[str, str], Path, dict]],
    stations: dict,
    output_hdf5: Path,
    output_csv: Path,
    target_rate: float,
    target_samples: int,
    pick_policy: str,
) -> dict:
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = defaultdict(int)
    csv_rows: List[dict] = []

    with h5py.File(str(output_hdf5), "w") as h5:
        group = h5.create_group("data")
        for key, station_dir, label in matched_items:
            try:
                data, waveform_meta = read_waveform_group(station_dir, target_rate, target_samples)
            except Exception as exc:
                summary["skipped_waveform_errors"] += 1
                print("Skipping {}: {}".format(station_dir, exc), file=sys.stderr)
                continue

            station, window_name = key
            network = waveform_meta["network"] or label.get("network", "")
            receiver_type = waveform_meta["receiver_type"] or label.get("label_channel", "")[:2]
            start_time = waveform_meta["trace_start_time"]
            trace_time = window_name.split("_", 1)[0].replace("T", "").replace("Z", "")
            trace_name = "{}.{}_{}_{}_EV".format(station, network, trace_time, receiver_type)

            p_sample = sample_index(label.get("p"), start_time, target_rate, target_samples)
            s_sample = sample_index(label.get("s"), start_time, target_rate, target_samples)
            if p_sample is None:
                summary["missing_or_out_of_window_p"] += 1
            if s_sample is None:
                summary["missing_or_out_of_window_s"] += 1
            if pick_policy == "both" and (p_sample is None or s_sample is None):
                summary["skipped_missing_required_picks"] += 1
                continue
            if p_sample is None and s_sample is None:
                summary["skipped_no_valid_picks"] += 1
                continue

            coda = coda_end_sample(p_sample, s_sample, target_samples)
            snr = estimate_snr_db(data, p_sample, s_sample)
            lat, lon, elv = station_coords(stations, station)

            dataset = group.create_dataset(trace_name, data=data, dtype=np.float32)
            dataset.attrs["trace_name"] = trace_name
            dataset.attrs["trace_category"] = "earthquake_local"
            dataset.attrs["trace_start_time"] = blockly_time(start_time)
            dataset.attrs["network_code"] = network
            dataset.attrs["receiver_code"] = station
            dataset.attrs["receiver_type"] = receiver_type
            dataset.attrs["receiver_latitude"] = h5_attr_float(lat)
            dataset.attrs["receiver_longitude"] = h5_attr_float(lon)
            dataset.attrs["receiver_elevation_m"] = h5_attr_float(elv)
            dataset.attrs["coda_end_sample"] = int(coda)
            dataset.attrs["snr_db"] = snr
            dataset.attrs["p_pn_pg_s_sn_sg"] = np.array(
                [
                    float(p_sample) if p_sample is not None else np.nan,
                    np.nan,
                    np.nan,
                    float(s_sample) if s_sample is not None else np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            )
            dataset.attrs["source_id"] = window_name
            dataset.attrs["eqcct_label_start_time"] = label["label_start_time"].isoformat()
            dataset.attrs["eqcct_label_end_time"] = label["label_end_time"].isoformat()
            if label.get("p") is not None:
                dataset.attrs["p_arrival_time"] = label["p"].isoformat()
            if label.get("s") is not None:
                dataset.attrs["s_arrival_time"] = label["s"].isoformat()
            if p_sample is not None:
                dataset.attrs["p_arrival_sample"] = int(p_sample)
                dataset.attrs["p_status"] = "manual"
                dataset.attrs["p_weight"] = 0.5
            if s_sample is not None:
                dataset.attrs["s_arrival_sample"] = int(s_sample)
                dataset.attrs["s_status"] = "manual"
                dataset.attrs["s_weight"] = 0.5

            csv_rows.append(
                {
                    "network_code": network,
                    "receiver_code": station,
                    "receiver_type": receiver_type,
                    "receiver_latitude": finite_or_blank(lat),
                    "receiver_longitude": finite_or_blank(lon),
                    "receiver_elevation_m": finite_or_blank(elv),
                    "p_arrival_sample": finite_or_blank(p_sample),
                    "p_status": "manual" if p_sample is not None else "",
                    "p_weight": 0.5 if p_sample is not None else "",
                    "p_travel_sec": "",
                    "s_arrival_sample": finite_or_blank(s_sample),
                    "s_status": "manual" if s_sample is not None else "",
                    "s_weight": 0.5 if s_sample is not None else "",
                    "source_id": window_name,
                    "source_origin_time": "",
                    "source_origin_uncertainty_sec": "",
                    "source_latitude": "",
                    "source_longitude": "",
                    "source_error_sec": "",
                    "source_gap_deg": "",
                    "source_horizontal_uncertainty_km": "",
                    "source_depth_km": "",
                    "source_depth_uncertainty_km": "",
                    "source_magnitude": "",
                    "source_magnitude_type": "",
                    "source_magnitude_author": "",
                    "source_mechanism_strike_dip_rake": "",
                    "source_distance_deg": "",
                    "source_distance_km": "",
                    "back_azimuth_deg": "",
                    "snr_db": snr_to_text(snr),
                    "coda_end_sample": coda,
                    "trace_start_time": blockly_time(start_time),
                    "trace_category": "earthquake_local",
                    "trace_name": trace_name,
                }
            )
            summary["written"] += 1

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    return dict(summary)


def verify_outputs(output_hdf5: Path, output_csv: Path, target_samples: int) -> dict:
    summary = defaultdict(int)
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    with h5py.File(str(output_hdf5), "r") as h5:
        if "data" not in h5:
            raise ValueError("HDF5 file has no top-level 'data' group")
        group = h5["data"]
        summary["csv_rows"] = len(rows)
        summary["hdf5_datasets"] = len(group)
        if len(rows) != len(group):
            raise ValueError("CSV rows ({}) != HDF5 datasets ({})".format(len(rows), len(group)))

        for row in rows:
            trace_name = row["trace_name"]
            if trace_name not in group:
                raise ValueError("Missing HDF5 dataset for CSV trace_name {}".format(trace_name))
            dataset = group[trace_name]
            if dataset.shape != (target_samples, 3):
                raise ValueError("{} has shape {}, expected ({}, 3)".format(trace_name, dataset.shape, target_samples))
            if dataset.dtype != np.dtype("float32"):
                raise ValueError("{} has dtype {}, expected float32".format(trace_name, dataset.dtype))
            attrs = dataset.attrs
            if attrs.get("trace_name") != trace_name:
                raise ValueError("{} trace_name attribute mismatch".format(trace_name))
            if attrs.get("trace_category") != "earthquake_local":
                raise ValueError("{} trace_category is not earthquake_local".format(trace_name))
            arrivals = np.asarray(attrs.get("p_pn_pg_s_sn_sg"))
            if arrivals.shape != (6,):
                raise ValueError("{} p_pn_pg_s_sn_sg must have six entries".format(trace_name))

            valid_pick_count = 0
            for attr_name in ["p_arrival_sample", "s_arrival_sample"]:
                if attr_name in attrs:
                    pick = int(attrs[attr_name])
                    if pick <= 0 or pick >= target_samples:
                        raise ValueError("{} {} out of range: {}".format(trace_name, attr_name, pick))
                    valid_pick_count += 1
            if valid_pick_count == 0:
                raise ValueError("{} has no valid P or S arrival sample".format(trace_name))
            summary["verified"] += 1

    return dict(summary)


def ensure_outputs_can_be_written(paths: Iterable[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists():
            if not overwrite:
                raise FileExistsError("{} already exists; pass --overwrite to replace it".format(path))
            path.unlink()


def main() -> int:
    args = parse_args()
    ensure_outputs_can_be_written([args.output_hdf5, args.output_csv], args.overwrite)

    labels, warnings, label_rows = load_labels(args.labels)
    waveform_groups = index_waveform_groups(args.waveform_dir)
    stations = load_station_metadata(args.stations_json)

    matched_items = []
    for key in sorted(waveform_groups):
        label = labels.get(key)
        if label is None:
            continue
        matched_items.append((key, waveform_groups[key], label))

    if args.max_records is not None:
        matched_items = matched_items[: args.max_records]

    unmatched_label_count = sum(1 for key in labels if key not in waveform_groups)
    duplicate_label_groups = sum(1 for value in labels.values() if value["rows"] > 1)

    print("Label rows read: {}".format(label_rows))
    print("Merged label groups: {}".format(len(labels)))
    print("Waveform station-window groups: {}".format(len(waveform_groups)))
    print("Matched groups to convert: {}".format(len(matched_items)))
    print("Unmatched label groups: {}".format(unmatched_label_count))
    print("Merged duplicate label groups: {}".format(duplicate_label_groups))
    if warnings:
        print("Warnings: {}".format(len(warnings)), file=sys.stderr)
        for warning in warnings[:20]:
            print("  {}".format(warning), file=sys.stderr)
        if len(warnings) > 20:
            print("  ... {} more warnings".format(len(warnings) - 20), file=sys.stderr)

    if not matched_items:
        print("No matched waveform/label groups found.", file=sys.stderr)
        return 2

    summary = write_outputs(
        matched_items=matched_items,
        stations=stations,
        output_hdf5=args.output_hdf5,
        output_csv=args.output_csv,
        target_rate=args.target_sampling_rate,
        target_samples=args.target_samples,
        pick_policy=args.pick_policy,
    )
    print("Conversion summary: {}".format(dict(sorted(summary.items()))))
    print("Wrote HDF5: {}".format(args.output_hdf5))
    print("Wrote CSV: {}".format(args.output_csv))

    if args.verify:
        verification = verify_outputs(args.output_hdf5, args.output_csv, args.target_samples)
        print("Verification summary: {}".format(dict(sorted(verification.items()))))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
