# Future-Mask Encoder-Decoder Trial Report

Smoke trial: 100 updates per condition on WikiText-2 with a small encoder-decoder transformer (2 enc / 2 dec layers, 128-dim, 8 heads, T=128, k=T/2, fp16/cpu fallback, --no-token-positional-embeddings).

This is a wiring-validation run, not a scientific result. Real evaluation needs the 8-layer/1024-dim config on WikiText-103.


## Results

| Cond | Spec | Train loss | Val loss | Val PPL | Probe MAE | Probe Acc |
| ---- | ---- | ----------:| --------:| -------:| ---------:| ---------:|
| 8C | `C,C,C,C,C,C,C,C` | 10.5620 | 10.1880 | 1166.25 | 23.26 | 0.019 |
| 4F4C | `F,F,F,F,C,C,C,C` | 10.5630 | 10.1870 | 1165.62 | 21.53 | 0.020 |
| 8B | `(empty: bidirectional)` | 10.5620 | 10.1870 | 1165.60 | 22.73 | 0.015 |

## Notes

- `Train loss` is the final logged `train_inner` cross-entropy.
- `Val loss` / `Val PPL` are from the post-training validation pass.
- `Probe MAE` is the mean absolute error of a 100-step linear probe trained on the frozen final-layer encoder hidden states to predict absolute position. Lower = better positional information.
- `Probe Acc` is the exact-position-match accuracy of that probe.
- 8B = fully bidirectional encoder (no mask).
- 8C = all heads causal.
- 4F4C = first 4 heads future-only, last 4 heads causal.

## Interpretation

Val PPLs are nearly identical across conditions (within 1 ppl out of ~1166) because 100 updates on a 2-layer/128-dim model is far too few to differentiate mask variants on the LM task — the model is still near uniform-over-vocab. The probe MAEs do separate slightly: 4F4C (future + causal mix) gives the lowest MAE, suggesting the mixed-mask encoder does encode position more recoverably than a pure-causal (8C) or pure-bidirectional (8B) encoder even at this scale. Treat this as a directional hint only; a real evaluation needs WikiText-103 and the 8-layer/1024-dim config.
