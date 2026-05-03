# Finnish EQTransformer Fine-Tuning Process

This guide documents the exact workflow for improving the pretrained EQTransformer model with the Finnish earthquake fine-tuning dataset in this repository, then comparing the pretrained model against the fine-tuned validation-best checkpoint.

Use the same held-out Finnish test split for both models. Do not compare results from different `test.npy` files.

## Data

Use the Finnish earthquake fine-tuning pair:

```text
data/finnish_eq_finetune.hdf5
data/finnish_eq_finetune.csv
```

The CSV must contain `trace_name`, and each trace must exist in the HDF5 file at:

```text
/data/{trace_name}
```

The expected waveform shape is:

```text
(6000, 3)
```

For P/S fine-tuning, each event trace should include `p_arrival_sample` and `s_arrival_sample` attributes.

## Start The Notebook

From the repository root:

```bash
cd notebook
jupyter notebook
```

Open:

```text
blocklyeqt.ipynb
```

## Fine-Tune With Finnish Data

In the notebook, set **Manage Datasets**:

```text
Upload HDF: ../data/finnish_eq_finetune.hdf5
Upload CSV: ../data/finnish_eq_finetune.csv
```

Set **Installed Models**:

```text
Upload Model: ../pretrained/EqT_model.h5
```

Set **Training Configs**:

```text
Mode: Fine Tuning
Train-Test Split Ratio (%): 80
Drop Rate: 0.1
Input Dimension: (6000,3)
Phase Types: Detector, P, S
Batch Size: 32
Epochs: 200
Patience: 40
Output Name: eq_finetune
```

Click **Launch Model**.

The training run creates the held-out split, training report, and checkpoint folder:

```text
notebook/eq_finetune_outputs/
notebook/eq_finetune_outputs/test.npy
notebook/eq_finetune_outputs/X_report.txt
notebook/eq_finetune_outputs/models/
```

## Use Only The Best Checkpoint

Use the best validation checkpoint saved in the training output `models/` folder:

```text
notebook/eq_finetune_outputs/models/<best_checkpoint>.h5
```

The checkpoint name depends on the output name, stopping point, and validation-loss history of that run. With the repository defaults, checkpoints are saved only when validation loss improves, so choose the checkpoint corresponding to the best validation epoch. If you keep only one checkpoint after cleanup, use that remaining `.h5` file.

The pretrained baseline remains:

```text
pretrained/EqT_model.h5
```

## Evaluate Pretrained Baseline

In **Validation**, evaluate the original pretrained model on the Finnish held-out split:

```text
Model: ../pretrained/EqT_model.h5
HDF5: ../data/finnish_eq_finetune.hdf5
Testset: eq_finetune_outputs/test.npy
Output: finnish_pretrained_eval
```

This produces:

```text
notebook/finnish_pretrained_eval_outputs/X_test_results.csv
notebook/finnish_pretrained_eval_outputs/X_report.txt
```

## Evaluate Fine-Tuned Best Checkpoint

In **Validation**, evaluate only the best checkpoint:

```text
Model: eq_finetune_outputs/models/<best_checkpoint>.h5
HDF5: ../data/finnish_eq_finetune.hdf5
Testset: eq_finetune_outputs/test.npy
Output: finnish_finetuned_best_eval
```

This produces:

```text
notebook/finnish_finetuned_best_eval_outputs/X_test_results.csv
notebook/finnish_finetuned_best_eval_outputs/X_report.txt
```

## Important Validation Setting

The notebook validation cell currently sets:

```python
number_of_plots=10
```

For metric calculation, use a value larger than the test set size so every detected trace is written to `X_test_results.csv`:

```python
number_of_plots=10000
```

Alternatively, run this cell directly from the notebook directory:

```python
from BlocklyEQTransformer.core.tester import tester

common = dict(
    input_hdf5="../data/finnish_eq_finetune.hdf5",
    input_testset="eq_finetune_outputs/test.npy",
    detection_threshold=0.20,
    P_threshold=0.1,
    S_threshold=0.1,
    number_of_plots=10000,
    estimate_uncertainty=False,
    number_of_sampling=2,
    loss_types=[
        "binary_crossentropy",
        "binary_crossentropy",
        "binary_crossentropy",
    ],
    loss_weights=[0.05, 0.40, 0.55],
    input_dimention=(6000, 3),
    normalization_mode="std",
    mode="generator",
    batch_size=10,
    gpuid=None,
    gpu_limit=None,
    phase_types=["d", "P", "S"],
)

tester(
    input_model="../pretrained/EqT_model.h5",
    output_name="finnish_pretrained_eval",
    **common,
)

tester(
    input_model="eq_finetune_outputs/models/<best_checkpoint>.h5",
    output_name="finnish_finetuned_best_eval",
    **common,
)
```

## Aggregate Comparison Metrics

The repository tester writes per-trace picks and errors, but it does not aggregate metrics. Use this after both evaluations finish:

```python
import csv
import math
import os
import numpy as np

base = "."
test = np.load(os.path.join(base, "eq_finetune_outputs", "test.npy"), allow_pickle=True)
total = len(test)

paths = {
    "pretrained": os.path.join(
        base,
        "finnish_pretrained_eval_outputs",
        "X_test_results.csv",
    ),
    "finetuned_best_checkpoint": os.path.join(
        base,
        "finnish_finetuned_best_eval_outputs",
        "X_test_results.csv",
    ),
}

for name, path in paths.items():
    with open(path) as handle:
        reader = csv.reader(handle)
        next(reader)
        rows = list(reader)

    detected = {
        row[4]
        for row in rows
        if len(row) > 14 and row[14] not in ("", "0", "None")
    }

    print(f"\n{name}")
    print(f"test_total: {total}")
    print(f"matched_event_recall_pct: {100 * len(detected) / total:.2f}")

    for phase, error_column in [("P", 20), ("S", 24)]:
        errors = []
        for row in rows:
            if len(row) > error_column and row[error_column] not in ("", "None"):
                errors.append(float(row[error_column]))

        errors = np.array(errors, dtype=float)
        print(f"{phase}_coverage_pct: {100 * len(errors) / total:.2f}")

        if len(errors):
            abs_errors = np.abs(errors)
            print(f"{phase}_mae_s: {abs_errors.mean() / 100:.4f}")
            print(f"{phase}_median_abs_s: {np.median(abs_errors) / 100:.4f}")
            print(f"{phase}_rmse_s: {math.sqrt(np.mean(errors ** 2)) / 100:.4f}")
            print(f"{phase}_within_0.1s_pct: {100 * np.mean(abs_errors <= 10):.2f}")
            print(f"{phase}_within_0.2s_pct: {100 * np.mean(abs_errors <= 20):.2f}")
            print(f"{phase}_within_0.5s_pct: {100 * np.mean(abs_errors <= 50):.2f}")
```

## Existing Run Results

These metrics were computed on:

```text
notebook/eq_finetune_outputs/test.npy
```

The held-out split contains 153 Finnish earthquake traces and no `_NO` noise traces. Because there are no noise traces, this split supports event recall and pick-error metrics, but it does not support detection precision or false-positive-rate estimates.

| Model | Matched event recall | P coverage | P MAE | S coverage | S MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained `../pretrained/EqT_model.h5` | 25.49% | 25.49% | 0.1064 s | 22.88% | 0.3111 s |
| Fine-tuned best checkpoint from this run | 96.08% | 92.81% | 0.0077 s | 2.61% | 0.1150 s |

The best checkpoint substantially improves event matching and P picking on this Finnish split. S-pick coverage is low, so further work should focus on S-phase performance before operational use.
