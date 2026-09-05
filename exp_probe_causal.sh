#!/bin/bash
#SBATCH --job-name=fixattn
#SBATCH --output=fixattn_%j.out
#SBATCH --error=fixattn_%j.err
#SBATCH --gpus=h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-long

set -euo pipefail

# =============================================================================
# -- ERROR TRAPPING ------------------------------------------------------------
# =============================================================================
CURRENT_STAGE="initialisation"

error_handler() {
    local exit_code=$?
    local line_no=$1
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  FATAL ERROR -- job aborted"
    echo "  Stage     : ${CURRENT_STAGE}"
    echo "  Line      : ${line_no}"
    echo "  Exit code : ${exit_code}"
    echo "  Command   : ${BASH_COMMAND}"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
}

trap 'error_handler ${LINENO}' ERR

# =============================================================================
# -- USER CONFIGURATION -------------------------------------------------------
# =============================================================================

# ---- Fixed-attention decoder architecture -----------------------------------
DECODER_LAYERS=2
DECODER_HEADS=1
DECODER_EMBED_DIM=512
DECODER_FFN_DIM=2048
TOKENS_PER_SAMPLE=512

# ---- Seed (encoded in paths for reproducibility) ---------------------------
SEED=999

# ---- Stage A: base LM training ---------------------------------------------
# The probe reads a *pretrained* decoder, so we first train a plain NoPos LM
# with exactly the architecture `fixed_attn_probe` rebuilds (the probe loads it
# with strict=True, so every arch flag below must match the probe's).
# This stage is the long pole of the job.
BASE_MAX_UPDATES=20000
BASE_MAX_TOKENS=8192
BASE_LR=5e-4

# ---- Probe training --------------------------------------------------------
PROBE_MAX_UPDATES=5000
PROBE_LR=5e-3
PROBE_MAX_TOKENS=4096
PROBE_VALIDATE_EVERY=500

# ---- Conditions to run ------------------------------------------------------
# Each condition: NAME|EXTRA_FLAGS
#   NAME        : identifier for result dirs / tags
#   EXTRA_FLAGS : additional fairseq flags appended to the train command
#
# causal    = default causal self-attention mask (position encoded by
#             cumulative average over causally-visible tokens)
#
# NOTE: a "nocausal" (bidirectional-mask) arm is NOT available yet. The per-head
# mask spec lives on the nested EncDecBaseConfig, not on the flat
# TransformerLanguageModelConfig that fixed_attn_lm extends, so
# `--decoder-head-mask-spec B` is rejected by argparse. Adding it needs a
# `maskconfig` field on FixedAttnLanguageModelConfig, mirroring
# transformer_lm_position_probe.py.
CONDITIONS=(
    "causal|"
    "nocausal|--decoder-head-mask-spec B"
)

# ---- Sequence lengths to evaluate ------------------------------------------
EVAL_LENGTHS=(512)

# ---- Probe layers (0 = embedding output, 1..N = decoder layer outputs) -----
# For a 2-layer decoder: 0 = embedding, 1 = layer 0, 2 = layer 1
PROBE_LAYERS=(0 1 2)

# ---- Paths ------------------------------------------------------------------
REPO_DIR="${SLURM_SUBMIT_DIR}"
DATA_RAW="${REPO_DIR}/wt103-raw/wikitext-103"
DATABIN="${REPO_DIR}/data-bin/wikitext-103"
CHECKPOINTS_ROOT="${REPO_DIR}/checkpoints_fixed_attn"
RESULTS_DIR="${REPO_DIR}/fixed_attn_probe_results"
BASE_LM_DIR="${CHECKPOINTS_ROOT}/base_lm_seed${SEED}"
BASE_CKPT="${BASE_LM_DIR}/checkpoint_last.pt"

echo "======================================================"
echo "  Fixed-Attention Causal Probe Experiment"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Node         : $(hostname)"
echo "  Conditions   : ${#CONDITIONS[@]}"
echo "  Probe layers : ${PROBE_LAYERS[*]}"
echo "  Eval lengths : ${EVAL_LENGTHS[*]}"
echo "  Seed         : ${SEED}"
echo "  Repo         : ${REPO_DIR}"
echo "======================================================"

# =============================================================================
# -- CUDA / GPU diagnostics ---------------------------------------------------
# =============================================================================
CURRENT_STAGE="GPU / CUDA detection"
echo "--- GPU info ---"
nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

CUDA_VER=$(nvidia-smi 2>/dev/null \
    | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' \
    | head -1 \
    | tr -d '.') || CUDA_VER=""
case "${CUDA_VER:-0}" in
    124|125|126) TORCH_CU="cu124" ;;
    121|122|123) TORCH_CU="cu121" ;;
    118|119|120) TORCH_CU="cu118" ;;
    *)           TORCH_CU="cu121"
                 echo "WARNING: could not detect CUDA version (got '${CUDA_VER}'), defaulting to cu121" ;;
esac
echo "PyTorch wheel: ${TORCH_CU}"
echo "---"

# =============================================================================
# Step 1: Python 3.10
# =============================================================================
CURRENT_STAGE="Step 1 -- Miniconda / Python 3.10 setup"

MINICONDA_DIR="${HOME}/miniconda3"

if [ ! -d "${MINICONDA_DIR}" ]; then
    echo "[1/5] Miniconda not found -- installing into ${MINICONDA_DIR} ..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda_install.sh
    bash /tmp/miniconda_install.sh -b -p "${MINICONDA_DIR}"
    rm /tmp/miniconda_install.sh
else
    echo "[1/5] Miniconda already present at ${MINICONDA_DIR}."
fi

source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

CONDA_ENV_NAME="nopos"
if conda env list | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
    echo "      Conda env '${CONDA_ENV_NAME}' already exists -- skipping creation."
else
    echo "      Creating conda env '${CONDA_ENV_NAME}' with Python 3.10 ..."
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y -c conda-forge --override-channels
fi

conda activate "${CONDA_ENV_NAME}"
PY="$(which python)"
echo "      Python: $("${PY}" --version)  at ${PY}"

"${PY}" -m pip install -q "pip<22.0.4"

# =============================================================================
# Step 2: Install dependencies
# =============================================================================
CURRENT_STAGE="Step 2 -- dependency installation (torch / fairseq)"
echo "[2/5] Installing dependencies ..."

"${PY}" -m pip install "torch>=2.5.0" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CU}"
"${PY}" -c "
import torch
print('torch    :', torch.__version__)
print('CUDA ok  :', torch.cuda.is_available())
print('CUDA ver :', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU      :', torch.cuda.get_device_name(0))
else:
    raise RuntimeError('CUDA not available')
"

"${PY}" -m pip install -q numpy datasets

cd "${REPO_DIR}"
mv pyproject.toml pyproject.toml.bak
echo "      Pulling latest code ..."
git checkout master 2>/dev/null || true
git pull --ff-only || echo "WARNING: git pull failed -- continuing with local code"
"${PY}" -m pip install -q -e . --no-build-isolation
mv pyproject.toml.bak pyproject.toml

"${PY}" -c "import fairseq; print('fairseq OK:', fairseq.__version__)"

# =============================================================================
# Step 3: Download & preprocess WikiText-103
# =============================================================================
CURRENT_STAGE="Step 3 -- WikiText-103 download / preprocessing"
echo "[3/5] Preparing WikiText-103 data ..."

if [ -d "${DATABIN}" ] && [ -n "$(ls -A "${DATABIN}" 2>/dev/null)" ]; then
    echo "      Preprocessed data already present -- skipping."
else
    echo "      Downloading WikiText-103 ..."
    mkdir -p "${DATA_RAW}"

    OUT="${DATA_RAW}" "${PY}" - <<'PYEOF'
import os
from datasets import load_dataset

out = os.environ["OUT"]
splits = [
    ("train",      "wiki.train.tokens"),
    ("validation", "wiki.valid.tokens"),
    ("test",       "wiki.test.tokens"),
]
print("Loading wikitext-103-v1 from HuggingFace ...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
for split, fname in splits:
    path = f"{out}/{fname}"
    with open(path, "w", encoding="utf-8") as f:
        for item in ds[split]:
            f.write(item["text"] + "\n")
    print(f"Wrote {path}")
print("Download complete.")
PYEOF

    echo "      Binarising with fairseq-preprocess ..."
    mkdir -p "${DATABIN}"
    "${PY}" -m fairseq_cli.preprocess \
        --only-source \
        --trainpref "${DATA_RAW}/wiki.train.tokens" \
        --validpref "${DATA_RAW}/wiki.valid.tokens" \
        --testpref  "${DATA_RAW}/wiki.test.tokens" \
        --destdir   "${DATABIN}" \
        --workers   4
    echo "      Preprocessing done."
fi

mkdir -p "${CHECKPOINTS_ROOT}"
mkdir -p "${RESULTS_DIR}"

# =============================================================================
# Step 4: Train the base NoPos LM that the probe reads
# =============================================================================
CURRENT_STAGE="Step 4 -- base LM training"

if [ -f "${BASE_CKPT}" ]; then
    echo "[4/5] Base LM checkpoint already present -- skipping."
    echo "      ${BASE_CKPT}"
else
    echo "[4/5] Training base NoPos LM ..."
    mkdir -p "${BASE_LM_DIR}"

    # fixed_attn_base_lm is a plain transformer_lm that shares fixed_attn_probe's
    # arch config function, so the two decoders cannot drift -- the probe loads
    # this checkpoint with strict=True. Do NOT swap in --arch transformer_lm: its
    # auto-registered arch function is a no-op, so base_lm_architecture would not
    # run and the LM would silently be post-norm while the probe is pre-norm.
    # Do not add --share-decoder-input-output-embed either (changes the key set).
    "${PY}" -m fairseq_cli.train "${DATABIN}" \
        --task                          language_modeling \
        --arch                          fixed_attn_base_lm \
        --criterion                     cross_entropy \
        --sample-break-mode             none \
        --tokens-per-sample             "${TOKENS_PER_SAMPLE}" \
        --decoder-layers                "${DECODER_LAYERS}" \
        --decoder-attention-heads       "${DECODER_HEADS}" \
        --decoder-embed-dim             "${DECODER_EMBED_DIM}" \
        --decoder-ffn-embed-dim         "${DECODER_FFN_DIM}" \
        --no-token-positional-embeddings \
        --dropout                       0.1 \
        --attention-dropout             0.0 \
        --optimizer                     adam \
        --adam-betas                    "(0.9, 0.98)" \
        --weight-decay                  0.0 \
        --clip-norm                     1.0 \
        --lr                            "${BASE_LR}" \
        --lr-scheduler                  inverse_sqrt \
        --warmup-updates                4000 \
        --max-tokens                    "${BASE_MAX_TOKENS}" \
        --max-update                    "${BASE_MAX_UPDATES}" \
        --skip-invalid-size-inputs-valid-test \
        --fp16 \
        --save-dir                      "${BASE_LM_DIR}" \
        --save-interval-updates         "${BASE_MAX_UPDATES}" \
        --keep-last-epochs              1 \
        --no-epoch-checkpoints \
        --log-interval                  200 \
        --log-format                    json \
        --num-workers                   4 \
        --seed                          "${SEED}"

    echo "      Base LM training done -- ${BASE_CKPT}"
fi

if [ ! -f "${BASE_CKPT}" ]; then
    echo "FATAL: base LM checkpoint missing at ${BASE_CKPT}" >&2
    exit 1
fi

# =============================================================================
# Step 5: Train position probes (layer x eval_length)
# =============================================================================
CURRENT_STAGE="Step 5 -- fixed-attention position probing"
echo "[5/5] Training fixed-attention position probes ..."

for cond_str in "${CONDITIONS[@]}"; do
    IFS='|' read -r COND_NAME COND_EXTRA <<< "${cond_str}"

    for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}_seed${SEED}_layer${LAYER_IDX}_len${EVAL_LEN}"
            SAVE_DIR="${CHECKPOINTS_ROOT}/${PROBE_TAG}"
            RESULT_FILE="${RESULTS_DIR}/${PROBE_TAG}.json"

            # Skip if result already exists
            if [ -f "${RESULT_FILE}" ]; then
                echo "      [${PROBE_TAG}] Result exists -- skipping."
                continue
            fi

            echo "      [${PROBE_TAG}] Training probe ..."
            mkdir -p "${SAVE_DIR}"

            TRAIN_ARGS=(
                "${DATABIN}"
                --task                          language_modeling_position_probe
                --arch                          fixed_attn_probe
                --criterion                     position_probe_ce
                --tokens-per-sample             "${EVAL_LEN}"
                --probe-layer-idx               "${LAYER_IDX}"
                --pretrained-decoder-filename   "${BASE_CKPT}"
                --decoder-layers                "${DECODER_LAYERS}"
                --decoder-attention-heads       "${DECODER_HEADS}"
                --decoder-embed-dim             "${DECODER_EMBED_DIM}"
                --decoder-ffn-embed-dim         "${DECODER_FFN_DIM}"
                --no-token-positional-embeddings
                --dropout                       0.0
                --attention-dropout             0.0
                --optimizer                     adam
                --adam-betas                    "(0.9, 0.98)"
                --weight-decay                  0.0
                --clip-norm                     1.0
                --lr                            "${PROBE_LR}"
                --lr-scheduler                  inverse_sqrt
                --warmup-updates                500
                --max-tokens                    "${PROBE_MAX_TOKENS}"
                --max-update                    "${PROBE_MAX_UPDATES}"
                --validate-interval-updates     "${PROBE_VALIDATE_EVERY}"
                --skip-invalid-size-inputs-valid-test
                --fp16
                --save-dir                      "${SAVE_DIR}"
                --save-interval-updates         "${PROBE_MAX_UPDATES}"
                --keep-last-epochs              1
                --no-epoch-checkpoints
                --log-interval                  50
                --log-format                    json
                --num-workers                   4
                --seed                          "${SEED}"
            )

            # Append condition-specific flags as individual array elements
            if [ -n "${COND_EXTRA}" ]; then
                # shellcheck disable=SC2206
                TRAIN_ARGS+=(${COND_EXTRA})
            fi

            TRAIN_LOG="${SAVE_DIR}/train.log"
            printf '      %s\n' "python -m fairseq_cli.train ${TRAIN_ARGS[*]}"
            # Tee so the summary step can recover accuracy / mad from the JSON log.
            "${PY}" -m fairseq_cli.train "${TRAIN_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"

            # Extract final validation metrics from the checkpoint + JSON log
            "${PY}" -c "
import json, os, sys

ckpt_path = os.path.join('${SAVE_DIR}', 'checkpoint_last.pt')
if not os.path.exists(ckpt_path):
    print(f'WARNING: {ckpt_path} not found', file=sys.stderr)
    sys.exit(0)

import torch
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
extra = state.get('extra_state', {})
val_loss = extra.get('best', float('nan'))

# --log-format json emits one JSON object per line; keep the last 'valid' record,
# which carries the accuracy and mean-absolute-difference the probe reports.
accuracy = mad = float('nan')
try:
    with open('${TRAIN_LOG}') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{') or 'valid' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if 'valid_accuracy' in rec:
                accuracy = float(rec['valid_accuracy'])
            if 'valid_mad' in rec:
                mad = float(rec['valid_mad'])
except OSError:
    pass

result = {
    'condition': '${COND_NAME}',
    'probe_layer': ${LAYER_IDX},
    'eval_length': ${EVAL_LEN},
    'seed': ${SEED},
    'val_loss': val_loss,
    'accuracy': accuracy,
    'mad': mad,
    'num_updates': state.get('optimizer_history', [{}])[-1].get('num_updates', -1),
}
with open('${RESULT_FILE}', 'w') as f:
    json.dump(result, f, indent=2)
print(f'      Saved ${RESULT_FILE}')
"

            echo "      [${PROBE_TAG}] Done."
        done
    done
done

# =============================================================================
# Step 6: Summarise results
# =============================================================================
CURRENT_STAGE="Summary"
echo ""
echo "======================================================"
echo "  FIXED-ATTENTION PROBE RESULTS"
echo "======================================================"
echo ""
echo "  Loss is in bits. Chance for ${TOKENS_PER_SAMPLE} positions is"
echo "  log2(${TOKENS_PER_SAMPLE} + 5) bits with accuracy ~0."
echo "  Layer 0 (embedding, no positional embeddings) MUST sit at chance."

for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
    echo ""
    echo "--- Eval sequence length: ${EVAL_LEN} ---"
    echo ""
    printf "%-15s  %5s  %8s  %8s  %8s  %10s\n" "CONDITION" "LAYER" "VAL_LOSS" "ACC(%)" "MAD" "UPDATES"
    printf "%-15s  %5s  %8s  %8s  %8s  %10s\n" "---------------" "-----" "--------" "--------" "--------" "----------"

    for cond_str in "${CONDITIONS[@]}"; do
        IFS='|' read -r COND_NAME COND_EXTRA <<< "${cond_str}"

        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}_seed${SEED}_layer${LAYER_IDX}_len${EVAL_LEN}"
            RESULT_FILE="${RESULTS_DIR}/${PROBE_TAG}.json"

            if [ -f "${RESULT_FILE}" ]; then
                read -r VAL_LOSS ACC MAD UPDATES <<< "$("${PY}" -c "
import json
d = json.load(open('${RESULT_FILE}'))
print(f\"{d['val_loss']:.4f}\",
      f\"{d.get('accuracy', float('nan')):.3f}\",
      f\"{d.get('mad', float('nan')):.2f}\",
      d['num_updates'])
")"
            else
                VAL_LOSS="N/A"; ACC="N/A"; MAD="N/A"; UPDATES="N/A"
            fi

            printf "%-15s  %5s  %8s  %8s  %8s  %10s\n" \
                "${COND_NAME}" "${LAYER_IDX}" "${VAL_LOSS}" "${ACC}" "${MAD}" "${UPDATES}"
        done
    done
done

echo ""
echo "======================================================"
echo "  Fixed-attention probe experiment complete."
echo "  Results : ${RESULTS_DIR}/"
echo "======================================================"
