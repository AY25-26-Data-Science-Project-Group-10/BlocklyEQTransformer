#!/usr/bin/env python3
"""Convert EQCCTPro explosion waveform folders to BlocklyEQTransformer HDF5/CSV.

This is intentionally separate from the earthquake converter because the
explosion waveform folders often contain multiple channel families for the
same station/window, such as BH, HH, LH, VH, CH, or SH. For consistency, this
script only uses complete HH E/N/Z triplets and skips station/windows without
HH data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from obspy import UTCDateTime, read

from convert_eqcctpro_to_blocklyeqt import (
    CSV_COLUMNS,
    blockly_time,
    coda_end_sample,
    compact_utc,
    component_column,
    ensure_outputs_can_be_written,
    estimate_snr_db,
    finite_or_blank,
    h5_attr_float,
    load_station_metadata,
    parse_label_file_name,
    parse_utc,
    sample_index,
    snr_to_text,
    station_coords,
    verify_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EQCCTPro explosion waveform folders and manual P/S labels "
            "into BlocklyEQTransformer fine-tuning HDF5 and CSV files."
        )
    )
    parser.add_argument(
        "--waveform-dir",
        default=Path("data/waveforms_explosions"),
        type=Path,
        help="Directory containing EQCCTPro explosion waveform windows.",
    )
    parser.add_argument(
        "--labels",
        default=[Path("data/ex_labels.csv")],
        nargs="+",
        type=Path,
        help="One or more EQCCTPro explosion label CSV files.",
    )
    parser.add_argument(
        "--stations-json",
        default=Path("station_list.json"),
        type=Path,
        help="Optional station metadata JSON with network/channels/coords.",
    )
    parser.add_argument(
        "--output-hdf5",
        default=Path("data/finnish_explosion_finetune.hdf5"),
        type=Path,
        help="Output HDF5 file.",
    )
    parser.add_argument(
        "--output-csv",
        default=Path("data/finnish_explosion_finetune.csv"),
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
        "--channel-family",
        default="HH",
        help="Required channel family to convert. Defaults to HH.",
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


def label_key(station: str, start: UTCDateTime, end: UTCDateTime) -> Tuple[str, str]:
    return station, "{}_{}".format(compact_utc(start), compact_utc(end))


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


def load_explosion_labels(label_paths: Sequence[Path]) -> Tuple[Dict[Tuple[str, str], dict], List[str], int]:
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

                key = label_key(parsed["station"], parsed["start_time"], parsed["end_time"])
                if key not in labels:
                    labels[key] = {
                        "key": key,
                        "network": parsed["network"],
                        "station": parsed["station"],
                        "location": parsed["location"],
                        "label_start_time": parsed["start_time"],
                        "label_end_time": parsed["end_time"],
                        "label_channels": [],
                        "p": None,
                        "s": None,
                        "pick_sources": {},
                        "conflicts": [],
                        "rows": 0,
                    }

                record = labels[key]
                record["rows"] += 1
                record["label_channels"].append(parsed["channel"])

                p_time = parse_utc(row.get("p_arrival_time"))
                if p_time is not None:
                    add_pick(record, "p", p_time, row, label_path)

                s_time = parse_utc(row.get("s_arrival_time"))
                if s_time is not None:
                    add_pick(record, "s", s_time, row, label_path)

    for record in labels.values():
        warnings.extend(record["conflicts"])

    return labels, warnings, rows_seen


def waveform_channel_parts(path: Path) -> Optional[Tuple[str, str, str, str]]:
    head = path.name.split("__", 1)[0]
    parts = head.split(".")
    if len(parts) < 4:
        return None
    network, station, location, channel = parts[:4]
    return network, station, location, channel


def index_waveform_groups(waveform_dir: Path) -> Dict[Tuple[str, str], dict]:
    groups: Dict[Tuple[str, str], dict] = {}
    for station_dir in sorted(waveform_dir.glob("*/*")):
        if not station_dir.is_dir():
            continue
        key = (station_dir.name, station_dir.parent.name)
        group = {
            "station_dir": station_dir,
            "families": defaultdict(dict),
            "networks": {},
            "receiver_types": {},
        }
        for mseed_path in sorted(station_dir.glob("*.mseed")):
            parsed = waveform_channel_parts(mseed_path)
            if parsed is None:
                continue
            network, _station, _location, channel = parsed
            component = channel[-1].upper()
            family = channel[:2].upper()
            if component_column(channel) is None:
                continue
            group["families"][family][component] = mseed_path
            group["networks"][family] = network
            group["receiver_types"][family] = family
        groups[key] = group
    return groups


def complete_families(group: dict) -> List[str]:
    complete = []
    for family, components in group["families"].items():
        if {"E", "N", "Z"}.issubset(set(components)):
            complete.append(family)
    return sorted(complete)


def choose_family(group: dict, required_family: str) -> Optional[Tuple[str, str]]:
    available = complete_families(group)
    required_family = required_family.upper()
    if required_family in available:
        return required_family, "required"
    return None


def read_family_waveform(
    group: dict,
    family: str,
    target_rate: float,
    target_samples: int,
) -> Tuple[np.ndarray, dict]:
    paths = group["families"][family]
    traces = {}
    for component, path in paths.items():
        stream = read(str(path))
        for trace in stream:
            col = component_column(trace.stats.channel)
            if col is not None:
                traces[col] = trace.copy()
                break

    missing_components = sorted(set([0, 1, 2]) - set(traces))
    if missing_components:
        raise ValueError(
            "missing components {} for family {} in {}".format(
                missing_components, family, group["station_dir"]
            )
        )

    common_start = max(trace.stats.starttime for trace in traces.values())
    target_end = common_start + (target_samples - 1) / target_rate
    data = np.zeros((target_samples, 3), dtype=np.float32)

    for col, trace in traces.items():
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

    z_trace = traces[2]
    metadata = {
        "network": z_trace.stats.network or group["networks"].get(family, ""),
        "station": z_trace.stats.station or group["station_dir"].name,
        "receiver_type": family,
        "trace_start_time": common_start,
    }
    return data, metadata


def write_outputs(
    matched_items: Sequence[Tuple[Tuple[str, str], dict, dict]],
    stations: dict,
    output_hdf5: Path,
    output_csv: Path,
    target_rate: float,
    target_samples: int,
    pick_policy: str,
    required_family: str,
) -> dict:
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = defaultdict(int)
    csv_rows: List[dict] = []

    with h5py.File(str(output_hdf5), "w") as h5:
        h5.attrs["source_dataset"] = "waveforms_explosions"
        h5.attrs["source_label_csv"] = "ex_labels.csv"
        group_out = h5.create_group("data")

        for key, waveform_group, label in matched_items:
            chosen = choose_family(waveform_group, required_family)
            if chosen is None:
                summary["skipped_no_required_family_triplet"] += 1
                continue
            family, family_selection = chosen

            try:
                data, waveform_meta = read_family_waveform(
                    waveform_group, family, target_rate, target_samples
                )
            except Exception as exc:
                summary["skipped_waveform_errors"] += 1
                print("Skipping {}: {}".format(waveform_group["station_dir"], exc), file=sys.stderr)
                continue

            station, window_name = key
            network = waveform_meta["network"] or label.get("network", "")
            receiver_type = waveform_meta["receiver_type"] or family
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

            dataset = group_out.create_dataset(trace_name, data=data, dtype=np.float32)
            dataset.attrs["trace_name"] = trace_name
            dataset.attrs["trace_category"] = "earthquake_local"
            dataset.attrs["source_type"] = "explosion"
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
            dataset.attrs["eqcct_label_channels"] = ",".join(label.get("label_channels", []))
            dataset.attrs["selected_channel_family"] = family
            dataset.attrs["channel_family_selection"] = family_selection
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
                    "source_magnitude_type": "explosion",
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
            summary["family_{}".format(family)] += 1
            summary["family_selection_{}".format(family_selection)] += 1

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    return dict(summary)


def main() -> int:
    args = parse_args()
    ensure_outputs_can_be_written([args.output_hdf5, args.output_csv], args.overwrite)

    labels, warnings, label_rows = load_explosion_labels(args.labels)
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
    required_family = args.channel_family.upper()
    required_family_groups = sum(
        1 for group in waveform_groups.values() if required_family in complete_families(group)
    )

    print("Label rows read: {}".format(label_rows))
    print("Merged label groups: {}".format(len(labels)))
    print("Waveform station-window groups: {}".format(len(waveform_groups)))
    print(
        "Waveform groups with complete {} triplets: {}".format(
            required_family, required_family_groups
        )
    )
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
        required_family=args.channel_family,
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
