# WikiText-103 Mixed Future-Mask Experiment

This experiment compares NoPos language models on WikiText-103 with:

- `0` future-masked heads: standard causal masking on all `8` heads.
- `1`, `2`, and `4` future-masked heads: the selected heads use the opposite mask.

Mask definitions follow the additive-attention convention requested for the experiment:

- Causal mask: for query `i` and key `j`, add `-inf` when `j > i`, else `0`.
- Future mask: for query `i` and key `j`, add `-inf` when `j < i`, else `0`.

The runner defaults to the WikiText-103 adaptive-input training recipe from
`examples/language_model/README.adaptive_inputs.md`, while also enabling
`--no-token-positional-embeddings` and explicitly fixing `--decoder-attention-heads 8`.

## Run

```bash
py -3 nopos_experiments/future_mask_wiki103/run_wiki103_future_mask_experiment.py
```

Useful options:

- `--python <path>`: use a different Python environment.
- `--vendor-dir <path>`: add a local dependency bundle to `PYTHONPATH`.
- `--no-user-site`: disable user-site packages, which is useful if another Python has a broken `torch` install in the user profile.
- `--fp16`: enable mixed precision when a CUDA-enabled PyTorch install is available.
- `--skip-preprocess`: reuse an existing `data-bin/wikitext-103`.
- `--summarize-only`: skip training and only regenerate the report from logs.
- `--future-head-counts 0,1,2,4`: change the compared head-count settings.

Outputs are written to `nopos_experiments/future_mask_wiki103/runs/`:

- `future_heads_<k>/train.log`
- `summary.json`
- `summary.csv`
- `report.md`
