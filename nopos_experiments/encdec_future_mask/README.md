# Encoder-Decoder Future-Mask Experiment

Extends NoPos with **per-head encoder self-attention masks** in an encoder-decoder
transformer. The encoder sees a length-`k` prefix; the decoder predicts the
length-`T-k` suffix. Per-head causal (`C`), future-only (`F`), and bidirectional
(`B`) masks let us test whether mixing mask types in the encoder improves
(a) language-model perplexity and (b) absolute position decodability from the
encoder hidden states.

## Code map

| File | Purpose |
| ---- | ------- |
| `fairseq/modules/multihead_attention.py` | accepts `[H, T, T]` per-head masks |
| `fairseq/models/transformer/encoder_head_mask.py` | builds `[H, T, T]` masks from spec strings |
| `fairseq/models/transformer/transformer_encoder.py` | wires the mask through encoder layers |
| `fairseq/models/transformer/transformer_config.py` | adds `--encoder-head-mask-spec` and `--encoder-future-mask-allow-self` |
| `fairseq/data/prefix_suffix_dataset.py` | splits each length-`T` block into (source, target) |
| `fairseq/tasks/encoder_decoder_language_modeling.py` | new task `encoder_decoder_language_modeling` |
| `nopos_experiments/encdec_future_mask/` | trial scripts + standalone probe |
| `tests/test_per_head_attn_mask.py` | per-head mask unit tests |

## Spec strings

`--encoder-head-mask-spec` is a comma-separated list of one token per encoder head:

- `C`: causal (head `i` attends to `j ≤ i`)
- `F`: future-only (head `i` attends to `j ≥ i` if `allow_self`, else `j > i`)
- `B`: bidirectional (no mask)

Empty string = no per-head mask = standard bidirectional encoder.

## Trial runner

```bash
export PYTHONPATH=$(pwd)
bash nopos_experiments/encdec_future_mask/run_trial.sh
```

Trains conditions `8C`, `4F4C`, `8B` for 100 updates each, then runs the
linear position probe and emits `REPORT.md`. Uses CUDA + fp16 if available,
otherwise falls back to CPU (much slower; smoke wiring only).

For a real evaluation, swap WikiText-2 for WikiText-103 and bump the model
to the paper's 8-layer / 1024-dim config with many more updates.

## GPU install

```
pip install torch                          # torch 2.11.0+cu130 is on PyPI
pip install -e .                           # installs fairseq runtime deps
```

The PyTorch wheel bundles its own CUDA runtime; only the NVIDIA driver
needs to be ≥ that version. CUDA-13 wheels run on any CUDA-13-capable
driver. If the cu130 wheel is unavailable, fall back to cu128 / cu126 /
cu124 — all work on a CUDA-13 driver.
