# Finnish EQTransformer Comparison Metrics

Comparison uses the same held-out Finnish earthquake test split:

```text
notebook/eq_finetune_outputs/test.npy
```

Test set size:

```text
193 earthquake traces
```

Models compared:

```text
Pretrained: pretrained/EqT_model.h5
Fine-tuned best checkpoint: notebook/eq_finetune_outputs/models/eq_finetune_056.h5
```

Raw evaluation outputs:

```text
notebook/finnish_pretrained_eval_outputs/X_test_results.csv
notebook/finnish_finetuned_best_eval_outputs/X_test_results.csv
```

## Summary Table

| Metric | Pretrained | Best checkpoint | Change |
| --- | ---: | ---: | ---: |
| Matched event recall | 15.03% | 89.12% | +74.09 percentage points |
| P coverage | 14.51% | 68.39% | +53.88 percentage points |
| P MAE | 0.1746 s | 0.1564 s | -0.0182 s |
| S coverage | 12.95% | 4.66% | -8.29 percentage points |
| S MAE | 0.4444 s | 0.0689 s | -0.3755 s |

## Interpretation

The best checkpoint substantially improves matched event recall and P coverage on the Finnish test split.

S-pick error is lower when an S pick is produced, but S coverage drops from 12.95% to 4.66%. Before operational use, further tuning should focus on S-phase coverage.

This split contains only earthquake traces and no `_NO` noise traces, so it cannot measure detection precision or false-positive rate.
