# WikiText-103 Mixed Future-Mask Experiment

Proxy CPU summary on full preprocessed WikiText-103 data with:
- `transformer_lm_wiki103`
- `8` decoder self-attention heads total
- `--no-token-positional-embeddings`
- `tokens-per-sample=512`
- `max-tokens=512`
- `update-freq=1`
- `max-update=100`

| Future-mask heads | Train loss | Last valid loss | Best valid loss | Best valid updates |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 18.9050 | 15.7670 | 15.7670 | 100 |
| 1 | 18.7660 | 15.1780 | 15.1780 | 100 |
| 2 | 18.7070 | 15.0040 | 15.0040 | 100 |
| 4 | 18.6390 | 14.9030 | 14.9030 | 100 |

## Conclusion

In this proxy setting, mixing future-masked heads with causal heads helped consistently. The best tested setting used `4` future-masked heads and improved validation loss by `0.8640` relative to the pure causal NoPos baseline.
