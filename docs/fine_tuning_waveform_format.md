# Fine-tuning waveform and label format

This project fine-tunes models with a pair of files:

- an HDF5 file containing waveform arrays and per-trace label metadata
- a CSV file listing the trace names to use from the HDF5 file

The reference sample files are:

- `ModelsAndSampleData/100samples.hdf5`
- `ModelsAndSampleData/100samples.csv`

The training code reads `input_csv` in `BlocklyEQTransformer/core/trainer.py`, takes the `trace_name` column, and loads each waveform from `input_hdf5` at `/data/{trace_name}`. Labels are not stored as separate arrays. They are generated during training from HDF5 dataset attributes such as `p_arrival_sample`, `s_arrival_sample`, and `p_pn_pg_s_sn_sg`.

## Required file structure

Use this layout:

```text
my_training_dataset/
|-- fine_tuning.hdf5
`-- fine_tuning.csv
```

Inside the HDF5 file:

```text
fine_tuning.hdf5
`-- data/
    |-- STATION.NET_YYYYMMDDHHMMSS_EV
    |-- STATION.NET_YYYYMMDDHHMMSS_EV
    `-- STATION.NET_YYYYMMDDHHMMSS_NO
```

Each item under `/data` must be an HDF5 dataset. The dataset name must exactly match one row in the CSV `trace_name` column.

## Waveform datasets

Each waveform dataset should contain a NumPy-compatible numeric array. The normal fine-tuning shape is:

```text
(6000, 3)
```

This means:

- 6000 samples per trace
- 3 components
- 100 Hz sampling rate for a 60-second window
- `float32` dtype is recommended

The expected component order for 3-component arrays is:

```text
column 0: E or 1 component
column 1: N or 2 component
column 2: Z component
```

The generator can also tolerate shorter traces, longer traces, one-component arrays, or arrays shaped as `(channels, samples)`. It pads, trims, or transposes internally. For fine-tuning, keep every trace as `(6000, 3)` unless you have a specific reason not to.

## Trace names

The code uses the suffix of `trace_name` to decide whether a trace is an event or noise:

- event traces must end with `_EV`
- noise traces must end with `_NO`

Examples:

```text
109C.TA_20060723155859_EV
RUF.HE_20240101123000_NO
```

The tester also parses station/network and receiver type from this pattern:

```text
{station}.{network}_{timestamp}_{receiver_type}_{EV-or-NO}
```

The sample dataset uses names like `109C.TA_20060723155859_EV`, which do not include a separate receiver type field. Training only requires the `_EV` or `_NO` suffix, but following the fuller pattern makes tester output metadata cleaner.

## CSV file

The minimum CSV file only needs a `trace_name` column:

```csv
trace_name
109C.TA_20060723155859_EV
RUF.HE_20240101123000_NO
```

The sample CSV contains many additional STEAD-style metadata columns. They are useful for inspection and reproducibility, but the trainer split logic only reads `trace_name`.

Recommended CSV columns are:

```text
network_code,receiver_code,receiver_type,receiver_latitude,receiver_longitude,
receiver_elevation_m,p_arrival_sample,p_status,p_weight,p_travel_sec,
s_arrival_sample,s_status,s_weight,source_id,source_origin_time,
source_origin_uncertainty_sec,source_latitude,source_longitude,
source_error_sec,source_gap_deg,source_horizontal_uncertainty_km,
source_depth_km,source_depth_uncertainty_km,source_magnitude,
source_magnitude_type,source_magnitude_author,
source_mechanism_strike_dip_rake,source_distance_deg,source_distance_km,
back_azimuth_deg,snr_db,coda_end_sample,trace_start_time,trace_category,
trace_name
```

If you include these columns, keep their values consistent with the HDF5 dataset attributes.

## Required HDF5 attributes for event traces

For each `_EV` dataset, include these attributes:

| Attribute | Type | Purpose |
| --- | --- | --- |
| `trace_name` | string | Same value as the dataset name and CSV `trace_name`. |
| `trace_category` | string | Use `earthquake_local` for event traces. |
| `p_arrival_sample` | int or float | P arrival sample index in the waveform window. |
| `s_arrival_sample` | int or float | S arrival sample index in the waveform window. |
| `coda_end_sample` | int, float, or small array | Used by augmentation in some modes. |
| `snr_db` | length-3 numeric array | Signal-to-noise ratio for the three components. Used by augmentation. |
| `trace_start_time` | string | Trace start time, usually `YYYY-MM-DD HH:MM:SS.ffffff`. |

The P and S sample indices are zero-based sample indices inside the waveform array. For the default `(6000, 3)` / 100 Hz / 60 s setup:

```text
arrival_sample = round((arrival_time - trace_start_time) * 100)
```

Valid arrival samples must be greater than `0` and less than or equal to `input_dimention[0]`. Values outside the training window are ignored by the generator.

## Optional HDF5 attributes

These attributes are not required to generate labels, but they are used by reports, plots, testing output, or data inspection:

```text
network_code
receiver_code
receiver_type
receiver_latitude
receiver_longitude
receiver_elevation_m
p_status
p_weight
p_travel_sec
s_status
s_weight
source_id
source_origin_time
source_origin_uncertainty_sec
source_latitude
source_longitude
source_error_sec
source_gap_deg
source_horizontal_uncertainty_km
source_depth_km
source_depth_uncertainty_km
source_magnitude
source_magnitude_type
source_magnitude_author
source_mechanism_strike_dip_rake
source_distance_deg
source_distance_km
back_azimuth_deg
```

For one-component datasets, you may add a `component` attribute so the loader can place the trace into the correct channel:

- `Z` maps to column 2
- `N` or `2` maps to column 1
- `E` or `0` maps to column 0

## Noise traces

Noise traces must end with `_NO` and should use:

```text
trace_category = "noise"
```

Noise traces do not need arrival attributes. Their picker and detector labels remain zero.

## Extra phases for transfer learning or fine-tuning

For phase sets beyond P and S, add this HDF5 attribute to each event trace:

```text
p_pn_pg_s_sn_sg = [P, Pn, Pg, S, Sn, Sg]
```

Use numeric sample indices for phases that exist and `numpy.nan` for missing phases.

Example:

```python
dataset.attrs["p_pn_pg_s_sn_sg"] = np.array([700, np.nan, 735, 1894, np.nan, 1930], dtype=np.float32)
```

The mapping is fixed:

```text
index 0: P
index 1: Pn
index 2: Pg
index 3: S
index 4: Sn
index 5: Sg
```

When `p_arrival_sample` and `s_arrival_sample` are also present, they override indices `0` and `3` for P and S.

## How labels are generated

During generator-mode training, `DataGenerator` creates output arrays shaped:

```text
(batch_size, input_dimention[0], 1)
```

The output names depend on `phase_types`:

- `d` creates `detector`
- `P` creates `picker_P`
- `S` creates `picker_S`
- `Pn` creates `picker_Pn`
- `Sn` creates `picker_Sn`
- `Pg` creates `picker_Pg`
- `Sg` creates `picker_Sg`

For `label_type="gaussian"`, each requested phase gets a Gaussian target centered on the arrival sample. The default phase window is 40 samples, so the nonzero label region is roughly arrival sample +/- 20 samples. The detector label is the sum of the requested picker labels.

Use `label_type="gaussian"` for fine-tuning, especially when training extra phase types such as `Pn`, `Pg`, `Sn`, and `Sg`. The older `triangle` and `box` label paths are P/S-oriented and should not be used for extra-phase fine-tuning.

## Fine-tuning call

Fine-tuning is selected with `retrain=2`. Transfer learning is `retrain=1`, which freezes most loaded-model layers. Fine-tuning loads the model but leaves layers trainable.

Example for P and S fine-tuning:

```python
from BlocklyEQTransformer.core.trainer import trainer

trainer(
    input_model="pretrained/EqT_model.h5",
    retrain=2,
    input_hdf5="my_training_dataset/fine_tuning.hdf5",
    input_csv="my_training_dataset/fine_tuning.csv",
    output_name="fine_tuned_model",
    input_dimention=(6000, 3),
    label_type="gaussian",
    mode="generator",
    phase_types=["d", "P", "S"],
    loss_types=["binary_crossentropy", "binary_crossentropy", "binary_crossentropy"],
    loss_weights=[0.05, 0.40, 0.55],
    batch_size=20,
    epochs=10,
)
```

Example for P, S, Pg, and Sg:

```python
trainer(
    input_model="pretrained/EqT_model.h5",
    retrain=2,
    input_hdf5="my_training_dataset/fine_tuning.hdf5",
    input_csv="my_training_dataset/fine_tuning.csv",
    output_name="fine_tuned_pg_sg_model",
    input_dimention=(6000, 3),
    label_type="gaussian",
    mode="generator",
    phase_types=["d", "P", "S", "Pg", "Sg"],
    loss_types=[
        "binary_crossentropy",
        "binary_crossentropy",
        "binary_crossentropy",
        "binary_crossentropy",
        "binary_crossentropy",
    ],
    loss_weights=[0.05, 0.40, 0.55, 0.40, 0.55],
)
```

Keep `loss_types` and `loss_weights` the same length and order as the model outputs created from `phase_types`. If `d` is included, detector loss comes first. Picker losses then follow the order of picker entries in `phase_types`.

## Minimal HDF5 and CSV construction example

```python
import csv
import h5py
import numpy as np

records = [
    {
        "trace_name": "RUF.HE_20240101123000_EV",
        "trace_category": "earthquake_local",
        "waveform": np.zeros((6000, 3), dtype=np.float32),
        "p_arrival_sample": 800,
        "s_arrival_sample": 1450,
        "coda_end_sample": 2500,
        "snr_db": np.array([20.0, 18.0, 22.0], dtype=np.float32),
        "trace_start_time": "2024-01-01 12:30:00.000000",
    },
    {
        "trace_name": "RUF.HE_20240101123100_NO",
        "trace_category": "noise",
        "waveform": np.zeros((6000, 3), dtype=np.float32),
        "trace_start_time": "2024-01-01 12:31:00.000000",
    },
]

with h5py.File("fine_tuning.hdf5", "w") as h5:
    group = h5.create_group("data")
    for record in records:
        trace_name = record["trace_name"]
        dataset = group.create_dataset(
            trace_name,
            data=record["waveform"],
            dtype=np.float32,
        )
        dataset.attrs["trace_name"] = trace_name
        dataset.attrs["trace_category"] = record["trace_category"]
        dataset.attrs["trace_start_time"] = record["trace_start_time"]

        if trace_name.endswith("_EV"):
            dataset.attrs["p_arrival_sample"] = record["p_arrival_sample"]
            dataset.attrs["s_arrival_sample"] = record["s_arrival_sample"]
            dataset.attrs["coda_end_sample"] = record["coda_end_sample"]
            dataset.attrs["snr_db"] = record["snr_db"]

with open("fine_tuning.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["trace_name"])
    writer.writeheader()
    for record in records:
        writer.writerow({"trace_name": record["trace_name"]})
```

## Converting the local EQCCTPro Finnish dataset

This repository includes a converter for the local EQCCTPro-style data under `data/waveforms_earthquakes`:

```bash
python scripts/convert_eqcctpro_to_blocklyeqt.py --overwrite --verify
```

By default it reads:

```text
data/waveforms_earthquakes/
data/eq_labels.csv
station_list.json
```

and writes:

```text
data/finnish_eq_finetune.hdf5
data/finnish_eq_finetune.csv
```

The converter merges duplicate EQCCTPro label rows for the same station/window, because some rows contain only a P pick and another row contains only an S pick. It resamples or trims MiniSEED traces as needed to produce `(6000, 3)` float32 arrays, computes sample labels relative to the output trace start time, and verifies that the generated CSV and HDF5 are loadable by the BlocklyEQTransformer training format.

By default the converter keeps only traces with both valid P and S picks, which is the safest choice when training `phase_types=["d", "P", "S"]` because the model does not mask unknown picks. To keep traces with only one valid pick, add:

```bash
--pick-policy any
```

For the local explosion waveform tree, use the separate HH-only converter:

```bash
python scripts/convert_eqcctpro_explosions_to_blocklyeqt.py --overwrite --verify
```

By default it reads `data/waveforms_explosions` and `data/ex_labels.csv`, writes `data/finnish_explosion_finetune.hdf5` and `data/finnish_explosion_finetune.csv`, and skips station/windows without a complete `HH` E/N/Z triplet.

## Validation checklist

Before training, check:

- `fine_tuning.hdf5` has a top-level `data` group.
- Every CSV `trace_name` exists as `/data/{trace_name}` in the HDF5 file.
- Event names end with `_EV`; noise names end with `_NO`.
- Event datasets have `trace_category="earthquake_local"`.
- Noise datasets have `trace_category="noise"`.
- Event datasets have valid `p_arrival_sample` and `s_arrival_sample` when training P and S pickers.
- Event datasets have `p_pn_pg_s_sn_sg` when training `Pn`, `Pg`, `Sn`, or `Sg`.
- Waveform arrays are preferably `(6000, 3)` and `float32`.
- Arrival sample indices are inside the waveform window.
- `phase_types`, `loss_types`, and `loss_weights` match the outputs you want to train.
