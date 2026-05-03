# Finnish EQTransformer Comparison Metrics

Comparison uses the same held-out Finnish earthquake test split:

```text
notebook/eq_finetune_outputs/test.npy
```

Test set size:

```text
153 earthquake traces
```

Models compared:

```text
Pretrained: pretrained/EqT_model.h5
Fine-tuned best checkpoint: notebook/eq_finetune_outputs/models/eq_finetune_062.h5
```

Raw evaluation outputs:

```text
notebook/finnish_pretrained_eval_outputs/X_test_results.csv
notebook/finnish_finetuned_best_eval_outputs/X_test_results.csv
```

## Summary Table

| Metric | Pretrained | Best checkpoint | Change |
| --- | ---: | ---: | ---: |
| Matched event recall | 25.49% | 96.08% | +70.59 percentage points |
| P coverage | 25.49% | 92.81% | +67.32 percentage points |
| P MAE | 0.1064 s | 0.0077 s | -0.0987 s |
| S coverage | 22.88% | 2.61% | -20.26 percentage points |
| S MAE | 0.3111 s | 0.1150 s | -0.1961 s |

## Interpretation

The best checkpoint substantially improves matched event recall and P picking on the Finnish test split.

S-pick error is lower when an S pick is produced, but S coverage drops from 22.88% to 2.61%. Before operational use, further tuning should focus on S-phase coverage.

This split contains only earthquake traces and no `_NO` noise traces, so it cannot measure detection precision or false-positive rate.
