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
notebook/eq_pretrained_raw_s_diagnostic.csv
notebook/eq_raw_s_diagnostic.csv
```

## Summary Table

| Metric | Pretrained | Best checkpoint |
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

## Interpretation

The best checkpoint substantially improves matched event recall and P coverage on the Finnish test split.

The original `X_test_results.csv` first-match S columns undercount useful S
picks because a later detection match can contain S even when the first written
match does not. The all-match and raw-S diagnostics show clear S-head
improvement after fine-tuning, though detector-window association still limits
accepted S coverage.

This split contains only earthquake traces and no `_NO` noise traces, so it cannot measure detection precision or false-positive rate.
