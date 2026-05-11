#!/bin/bash
# Train one encoder-decoder future-mask experiment condition.
# Usage: bash train_one.sh <condition_name> <head_spec> [extra fairseq-train args...]
#   <head_spec>: comma-separated per-head tokens (C/F/B), 8 entries, OR empty string ""
set -euo pipefail

COND="${1:?condition name required (e.g. 8C, 4F4C, 8B)}"
SPEC="${2-}"
shift 2 || true

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$ROOT"

DATA_BIN="data-bin/wikitext-2"
SAVE_DIR="checkpoints/encdec_${COND}"
mkdir -p "$SAVE_DIR"

# Detect CUDA. If not available, fall back to --cpu (smoke wiring only).
DEVICE_FLAGS="--fp16"
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[$COND] WARNING: CUDA not available, falling back to CPU (smoke only, much slower)" >&2
    DEVICE_FLAGS="--cpu"
fi

echo "[$COND] spec='$SPEC' device_flags='$DEVICE_FLAGS'"

# encoder_head_mask_spec is only set when non-empty; empty string = 8B (bidirectional).
SPEC_FLAG=()
if [ -n "$SPEC" ]; then
    SPEC_FLAG=(--encoder-head-mask-spec "$SPEC")
fi

python3 "$ROOT/fairseq_cli/train.py" "$DATA_BIN" \
    --task encoder_decoder_language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 128 \
    --encoder-prefix-fraction 0.5 \
    --arch transformer \
    --share-all-embeddings \
    --no-token-positional-embeddings \
    "${SPEC_FLAG[@]}" \
    --encoder-layers 2 --decoder-layers 2 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --encoder-embed-dim 128 --decoder-embed-dim 128 \
    --encoder-ffn-embed-dim 256 --decoder-ffn-embed-dim 256 \
    --dropout 0.1 --attention-dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 50 \
    --criterion cross_entropy \
    --max-tokens 2048 \
    --max-update 100 \
    --max-epoch 1 \
    --skip-invalid-size-inputs-valid-test \
    --required-batch-size-multiple 1 \
    --validate-interval-updates 100 \
    --save-interval-updates 100 \
    --log-interval 10 \
    --log-format json \
    --save-dir "$SAVE_DIR" \
    --seed 1 \
    --num-workers 0 \
    $DEVICE_FLAGS \
    "$@" \
    2>&1 | tee "$SAVE_DIR/train.log"
