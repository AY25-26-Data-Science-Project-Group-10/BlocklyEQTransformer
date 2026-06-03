# Finnish EQTransformer Fine-Tuning Process

This guide documents the exact workflow for improving the pretrained EQTransformer model with the Finnish earthquake and explosion fine-tuning datasets in this repository, then comparing the pretrained model against the fine-tuned validation-best checkpoint.

Use the same held-out Finnish test split for both models. Do not compare results from different `test.npy` files.

## Data

Use the Finnish earthquake fine-tuning pair:

```text
data/finnish_eq_finetune.hdf5
data/finnish_eq_finetune.csv
```

Use the Finnish explosion fine-tuning pair:

```text
data/finnish_explosion_finetune.hdf5
data/finnish_explosion_finetune.csv
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

Alternatively, run the same notebook as a browser-based Voila interface:

```bash
cd notebook
voila blocklyeqt.ipynb
```

Voila exposes the same fields as the notebook in a browser UI. Fill the fields described below, then click **Launch Model** to start fine-tuning or **Validate** to run evaluation. Paths are still relative to the `notebook/` directory.

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

The notebook validation callback should use:

```python
number_of_plots=10000
```

This value is intentionally larger than the Finnish test set size so every detected trace is written to `X_test_results.csv`. If an older copy of the notebook still has:

```python
number_of_plots=10
```

change it to `10000` before using Voila for comparison metrics.

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

The repository tester writes per-trace picks and errors. Aggregate the pretrained
and fine-tuned comparison metrics with:

```bash
python3 scripts/print_finnish_comparison.py
```

If the raw-S diagnostic CSVs need to be generated or refreshed, run the same
script from the Keras/TensorFlow environment with:

```bash
python3 scripts/print_finnish_comparison.py --compute-raw-s
```

By default, the script reads:

```text
notebook/eq_finetune_outputs/test.npy
notebook/finnish_pretrained_eval_outputs/X_test_results.csv
notebook/finnish_finetuned_best_eval_outputs/X_test_results.csv
notebook/eq_pretrained_raw_s_diagnostic.csv
notebook/eq_raw_s_diagnostic.csv
```

## Existing Run Results

These metrics were computed on:

```text
notebook/eq_finetune_outputs/test.npy
```

The held-out split contains 193 Finnish earthquake traces and no `_NO` noise traces. Because there are no noise traces, this split supports event recall and pick-error metrics, but it does not support detection precision or false-positive-rate estimates.

| Metric | Pretrained | Fine-tuned best |
| --- | ---: | ---: |
| Matched event recall | 15.03% | 89.12% |
| P coverage | 14.51% | 68.39% |
| P MAE | 0.1746 s | 0.1564 s |
| S coverage (CSV first match) | 12.95% | 4.66% |
| S MAE (CSV first match) | 0.4444 s | 0.0689 s |
| S coverage (all picker matches) | 13.47% | 49.22% |
| S MAE (all picker matches) | 0.4358 s | 0.0902 s |
| Raw S peak within 0.2 s | 5.18% | 53.89% |
| Raw S peak within 0.5 s | 9.33% | 71.50% |
| Raw S peak within 1.0 s | 11.92% | 84.46% |
| Raw S nearest-peak MAE | 0.4358 s | 1.3974 s |

The best checkpoint substantially improves event matching and P picking on this
Finnish split. The raw S head is also much better after fine-tuning. The
original `X_test_results.csv` first-match S columns undercount useful S picks
because a later detection match can contain S even when the first written match
does not.

## Fine-Tune With Finnish Explosion Data

Use the same notebook workflow and model settings as the earthquake run. Replace
only the data paths and output names:

```text
Upload HDF: ../data/finnish_explosion_finetune.hdf5
Upload CSV: ../data/finnish_explosion_finetune.csv
Output Name: expl_finetune
```

Use the best validation checkpoint saved in:

```text
notebook/expl_finetune_outputs/models/<best_checkpoint>.h5
```

## Evaluate Explosion Best Checkpoint

Use the same validation settings as the earthquake run, with explosion paths:

```text
Model: expl_finetune_outputs/models/<best_checkpoint>.h5
HDF5: ../data/finnish_explosion_finetune.hdf5
Testset: expl_finetune_outputs/test.npy
Output: finnish_expl_finetuned_best_eval
```

This produces:

```text
notebook/finnish_expl_finetuned_best_eval_outputs/X_test_results.csv
notebook/finnish_expl_finetuned_best_eval_outputs/X_report.txt
```

Optional pretrained explosion baseline:

```text
Model: ../pretrained/EqT_model.h5
HDF5: ../data/finnish_explosion_finetune.hdf5
Testset: expl_finetune_outputs/test.npy
Output: finnish_expl_pretrained_eval
```

This produces:

```text
notebook/finnish_expl_pretrained_eval_outputs/X_test_results.csv
notebook/finnish_expl_pretrained_eval_outputs/X_report.txt
```

## Aggregate Explosion Metrics

After the explosion validation result exists, aggregate the fine-tuned metrics
with:

```bash
python3 scripts/print_finnish_explosion_comparison.py
```

If the raw-S diagnostic CSVs need to be generated or refreshed, run:

```bash
python3 scripts/print_finnish_explosion_comparison.py --compute-raw-s
```

By default, the script reads:

```text
notebook/expl_finetune_outputs/test.npy
notebook/finnish_expl_finetuned_best_eval_outputs/X_test_results.csv
notebook/finnish_expl_pretrained_eval_outputs/X_test_results.csv
notebook/expl_pretrained_raw_s_diagnostic.csv
notebook/expl_raw_s_diagnostic.csv
```

If you also evaluated the pretrained explosion baseline, include it in the
comparison table:

```bash
python3 scripts/print_finnish_explosion_comparison.py \
  --pretrained notebook/finnish_expl_pretrained_eval_outputs/X_test_results.csv
```

## Existing Explosion Run Results

These metrics were computed on:

```text
notebook/expl_finetune_outputs/test.npy
```

The held-out split contains 416 Finnish explosion traces and no `_NO` noise
traces. Because there are no noise traces, this split supports event recall and
pick-error metrics, but it does not support detection precision or
false-positive-rate estimates.

| Metric | Pretrained | Fine-tuned best |
| --- | ---: | ---: |
| Matched event recall | 5.05% | 82.45% |
| P coverage | 5.05% | 70.91% |
| P MAE | 0.1652 s | 0.1653 s |
| S coverage (CSV first match) | 4.09% | 0.48% |
| S MAE (CSV first match) | 0.8612 s | 0.0800 s |
| S coverage (all picker matches) | 3.85% | 20.91% |
| S MAE (all picker matches) | 0.8538 s | 0.1031 s |
| Raw S peak within 0.2 s | 2.40% | 34.13% |
| Raw S peak within 0.5 s | 2.88% | 60.10% |
| Raw S peak within 1.0 s | 3.37% | 75.00% |
| Raw S nearest-peak MAE | 0.7994 s | 0.6343 s |

The explosion checkpoint substantially improves event matching and P coverage
on this split. The raw S head improves substantially, but all-match S coverage
still remains much lower than raw S near-peak coverage, so detector-window
association remains a limiting factor.
