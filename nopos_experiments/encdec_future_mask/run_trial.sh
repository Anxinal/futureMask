#!/bin/bash
# End-to-end smoke trial for 3 conditions: 8C, 4F4C, 8B.
# Trains each for 100 updates, then runs a 100-step linear position probe.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE"/../.. && pwd)"
cd "$ROOT"

bash "$HERE/prepare_data.sh"

# Map condition -> head-mask spec. Empty spec = bidirectional (8B baseline).
declare -A SPECS=(
    [8C]="C,C,C,C,C,C,C,C"
    [4F4C]="F,F,F,F,C,C,C,C"
    [8B]=""
    [4C]="C,C,C,C"
)

CONDS=(8C 4F4C 8B)

for c in "${CONDS[@]}"; do
    echo "=== Training condition $c (spec='${SPECS[$c]}') ==="
    bash "$HERE/train_one.sh" "$c" "${SPECS[$c]}"
done

for c in "${CONDS[@]}"; do
    CKPT="checkpoints/encdec_${c}/checkpoint_last.pt"
    if [ ! -f "$CKPT" ]; then
        echo "missing checkpoint: $CKPT" >&2
        continue
    fi
    echo "=== Probing condition $c ==="
    PROBE_FLAGS=""
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        PROBE_FLAGS="--cpu"
    fi
    python3 "$HERE/run_position_probe.py" \
        --checkpoint "$CKPT" \
        --data data-bin/wikitext-2 \
        --probe-updates 100 \
        --probe-layer -1 \
        --output "checkpoints/encdec_${c}/probe_metrics.json" \
        $PROBE_FLAGS
done

python3 "$HERE/summarize.py" \
    --conditions "${CONDS[@]}" \
    --out "$HERE/REPORT.md"
echo "wrote $HERE/REPORT.md"
