#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --output=probe_%j.out
#SBATCH --error=probe_%j.err
#SBATCH --nodelist=xgpi[0-20]
#SBATCH --gpus = a100-80: 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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

# ---- Architecture (shared across all conditions) ----------------------------
ENCODER_LAYERS=8
DECODER_LAYERS=8
ENCODER_HEADS=8
DECODER_HEADS=8
ENCODER_EMBED_DIM=1024
DECODER_EMBED_DIM=1024
ENCODER_FFN_DIM=4096
DECODER_FFN_DIM=4096
TOKENS_PER_SAMPLE=512
ENCODER_PREFIX_FRACTION=0.5

# ---- Phase 1: Base LM training ---------------------------------------------
LM_MAX_TOKENS=16384
LM_UPDATE_FREQ=1
LM_MAX_UPDATES=80000
LM_VALIDATE_EVERY=2000
LM_WARMUP=10000

# ---- Phase 2: Probe training -----------------------------------------------
PROBE_MAX_UPDATES=5000
PROBE_LR=5e-3
PROBE_MAX_TOKENS=4096
PROBE_VALIDATE_EVERY=500

# ---- Phase 2b: Relative position probe -------------------------------------
REL_PROBE_PAIRS_PER_SEQ=64
REL_PROBE_NUM_CLASSES=1024

# ---- Length extrapolation ---------------------------------------------------
# Eval sequence lengths: 512 = training length, 1024/2048 = extrapolation
EVAL_LENGTHS=(512 1024 2048)

# ---- Conditions to run ------------------------------------------------------
# Each condition: NAME|NOPOS_FLAG|ENCODER_MASK|EXTRA_FLAGS
#   NAME         : identifier for checkpoint dirs and logs
#   NOPOS_FLAG   : "yes" to disable positional embeddings, "no" to keep them
#   ENCODER_MASK : per-head encoder self-attention mask spec (empty = default)
#   EXTRA_FLAGS  : additional fairseq flags (e.g. --rotary-embedding)
#
# Conditions from experiment_instruction_refined.md + baselines:
CONDITIONS=(
    "pos_8B|no|B,B,B,B,B,B,B,B|"
    # "rope_8B|yes|B,B,B,B,B,B,B,B|--rotary-embedding"  # uncomment after verifying cluster has latest code
    "nopos_8B|yes|B,B,B,B,B,B,B,B|"
    "nopos_8C|yes|C,C,C,C,C,C,C,C|"
    "nopos_4F4C|yes|F,F,F,F,C,C,C,C|"
  
)

# ---- Probe layers to evaluate (0-indexed; -1 = last layer) -----------------
PROBE_LAYERS=(0 1 2 3 4 5 6 7)

# ---- Paths ------------------------------------------------------------------
REPO_DIR="${SLURM_SUBMIT_DIR}"
DATA_RAW="${REPO_DIR}/wt103-raw/wikitext-103"
DATABIN="${REPO_DIR}/data-bin/wikitext-103"
CHECKPOINTS_ROOT="${REPO_DIR}/checkpoints"
PROBE_RESULTS="${REPO_DIR}/probe_results"
REL_PROBE_RESULTS="${REPO_DIR}/rel_probe_results"
REG_REL_PROBE_RESULTS="${REPO_DIR}/reg_rel_probe_results"

echo "======================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Node         : $(hostname)"
echo "  Conditions   : ${#CONDITIONS[@]}"
echo "  Probe layers : ${PROBE_LAYERS[*]}"
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
    echo "[1/7] Miniconda not found -- installing into ${MINICONDA_DIR} ..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda_install.sh
    bash /tmp/miniconda_install.sh -b -p "${MINICONDA_DIR}"
    rm /tmp/miniconda_install.sh
else
    echo "[1/7] Miniconda already present at ${MINICONDA_DIR}."
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
echo "[2/7] Installing dependencies ..."

"${PY}" -m pip install --force-reinstall "torch>=2.7.0" \
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
"${PY}" -m pip install -q -e . --no-build-isolation
mv pyproject.toml.bak pyproject.toml

"${PY}" -c "import fairseq; print('fairseq OK:', fairseq.__version__)"

# =============================================================================
# Step 3: Download & preprocess WikiText-103
# =============================================================================
CURRENT_STAGE="Step 3 -- WikiText-103 download / preprocessing"
echo "[3/7] Preparing WikiText-103 data ..."

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

# =============================================================================
# Step 4: Train base language models (one per condition)
# =============================================================================
CURRENT_STAGE="Step 4 -- base LM training"
echo "[4/7] Training base language models ..."

for cond_str in "${CONDITIONS[@]}"; do
    IFS='|' read -r COND_NAME NOPOS_FLAG ENCODER_MASK COND_EXTRA <<< "${cond_str}"

    SAVE_DIR="${CHECKPOINTS_ROOT}/${COND_NAME}"
    CHECKPOINT="${SAVE_DIR}/checkpoint_last.pt"

    # Skip if checkpoint already exists
    if [ -f "${CHECKPOINT}" ]; then
        echo "      [${COND_NAME}] Checkpoint exists -- skipping LM training."
        continue
    fi

    echo "      [${COND_NAME}] Training enc-dec LM (nopos=${NOPOS_FLAG}, encoder_mask=${ENCODER_MASK:-default}) ..."
    mkdir -p "${SAVE_DIR}"

    # Build optional flags
    EXTRA_FLAGS=""
    if [ "${NOPOS_FLAG}" = "yes" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --no-token-positional-embeddings"
    fi
    if [ -n "${ENCODER_MASK}" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --encoder-head-mask-spec ${ENCODER_MASK}"
    fi
    if [ -n "${COND_EXTRA}" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} ${COND_EXTRA}"
    fi

    "${PY}" -m fairseq_cli.train "${DATABIN}" \
        --task                          encoder_decoder_language_modeling \
        --sample-break-mode             none \
        --tokens-per-sample             "${TOKENS_PER_SAMPLE}" \
        --encoder-prefix-fraction       "${ENCODER_PREFIX_FRACTION}" \
        --arch                          transformer \
        --encoder-layers                "${ENCODER_LAYERS}" \
        --decoder-layers                "${DECODER_LAYERS}" \
        --encoder-attention-heads       "${ENCODER_HEADS}" \
        --decoder-attention-heads       "${DECODER_HEADS}" \
        --encoder-embed-dim             "${ENCODER_EMBED_DIM}" \
        --decoder-embed-dim             "${DECODER_EMBED_DIM}" \
        --encoder-ffn-embed-dim         "${ENCODER_FFN_DIM}" \
        --decoder-ffn-embed-dim         "${DECODER_FFN_DIM}" \
        --share-all-embeddings \
        --dropout                       0.1 \
        --attention-dropout             0.1 \
        --optimizer                     adam \
        --adam-betas                    "(0.9, 0.98)" \
        --weight-decay                  0.01 \
        --clip-norm                     1.0 \
        --lr                            1e-4 \
        --lr-scheduler                  inverse_sqrt \
        --warmup-updates                "${LM_WARMUP}" \
        --criterion                     cross_entropy \
        --max-tokens                    "${LM_MAX_TOKENS}" \
        --update-freq                   "${LM_UPDATE_FREQ}" \
        --max-update                    "${LM_MAX_UPDATES}" \
        --skip-invalid-size-inputs-valid-test \
        --required-batch-size-multiple  1 \
        --validate-interval-updates     "${LM_VALIDATE_EVERY}" \
        --fp16 \
        --save-dir                      "${SAVE_DIR}" \
        --save-interval-updates         "${LM_MAX_UPDATES}" \
        --keep-last-epochs              1 \
        --no-epoch-checkpoints \
        --log-interval                  100 \
        --log-format                    json \
        --num-workers                   4 \
        --seed                          1 \
        ${EXTRA_FLAGS}

    echo "      [${COND_NAME}] Base LM training complete. Checkpoint: ${CHECKPOINT}"
done

# =============================================================================
# Step 5: Absolute position probes (condition x layer x eval_length)
# =============================================================================
CURRENT_STAGE="Step 5 -- absolute position probing"
echo "[5/7] Running absolute position probes ..."

mkdir -p "${PROBE_RESULTS}"

for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
    for cond_str in "${CONDITIONS[@]}"; do
        IFS='|' read -r COND_NAME NOPOS_FLAG ENCODER_MASK COND_EXTRA <<< "${cond_str}"

        CHECKPOINT="${CHECKPOINTS_ROOT}/${COND_NAME}/checkpoint_last.pt"
        if [ ! -f "${CHECKPOINT}" ]; then
            echo "      [${COND_NAME}] WARNING: checkpoint not found -- skipping."
            continue
        fi

        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}_layer${LAYER_IDX}_len${EVAL_LEN}"
            PROBE_OUT="${PROBE_RESULTS}/${PROBE_TAG}.json"

            if [ -f "${PROBE_OUT}" ]; then
                echo "      [${PROBE_TAG}] Result exists -- skipping."
                continue
            fi

            echo "      [${PROBE_TAG}] Training absolute position probe ..."

            "${PY}" "${REPO_DIR}/nopos_experiments/encdec_future_mask/run_position_probe.py" \
                --checkpoint            "${CHECKPOINT}" \
                --data                  "${DATABIN}" \
                --split                 valid \
                --train-split           train \
                --probe-layer           "${LAYER_IDX}" \
                --probe-updates         "${PROBE_MAX_UPDATES}" \
                --lr                    "${PROBE_LR}" \
                --max-tokens            "${PROBE_MAX_TOKENS}" \
                --eval-tokens-per-sample "${EVAL_LEN}" \
                --output                "${PROBE_OUT}"

            echo "      [${PROBE_TAG}] Done."
        done
    done
done

# =============================================================================
# Step 6: Relative position probes (condition x layer x eval_length)
# =============================================================================
CURRENT_STAGE="Step 6 -- relative position probing"
echo "[6/7] Running relative position probes ..."

mkdir -p "${REL_PROBE_RESULTS}"

for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
    for cond_str in "${CONDITIONS[@]}"; do
        IFS='|' read -r COND_NAME NOPOS_FLAG ENCODER_MASK COND_EXTRA <<< "${cond_str}"

        CHECKPOINT="${CHECKPOINTS_ROOT}/${COND_NAME}/checkpoint_last.pt"
        if [ ! -f "${CHECKPOINT}" ]; then
            echo "      [${COND_NAME}] WARNING: checkpoint not found -- skipping."
            continue
        fi

        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}_layer${LAYER_IDX}_len${EVAL_LEN}"
            REL_OUT="${REL_PROBE_RESULTS}/${PROBE_TAG}.json"

            if [ -f "${REL_OUT}" ]; then
                echo "      [${PROBE_TAG}] Result exists -- skipping."
                continue
            fi

            echo "      [${PROBE_TAG}] Training relative position probe ..."

            "${PY}" "${REPO_DIR}/nopos_experiments/encdec_future_mask/run_relative_position_probe.py" \
                --checkpoint            "${CHECKPOINT}" \
                --data                  "${DATABIN}" \
                --split                 valid \
                --train-split           train \
                --probe-layer           "${LAYER_IDX}" \
                --probe-updates         "${PROBE_MAX_UPDATES}" \
                --lr                    "${PROBE_LR}" \
                --max-tokens            "${PROBE_MAX_TOKENS}" \
                --num-rel-classes       "${REL_PROBE_NUM_CLASSES}" \
                --pairs-per-seq         "${REL_PROBE_PAIRS_PER_SEQ}" \
                --eval-tokens-per-sample "${EVAL_LEN}" \
                --output                "${REL_OUT}"

            echo "      [${PROBE_TAG}] Done."
        done
    done
done

# =============================================================================
# Step 6b: Relative position probes -- regression (condition x layer x eval_length)
# =============================================================================
CURRENT_STAGE="Step 6b -- relative position probing (regression)"
echo "[6b/7] Running relative position probes (regression) ..."

mkdir -p "${REG_REL_PROBE_RESULTS}"

for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
    for cond_str in "${CONDITIONS[@]}"; do
        IFS='|' read -r COND_NAME NOPOS_FLAG ENCODER_MASK COND_EXTRA <<< "${cond_str}"

        CHECKPOINT="${CHECKPOINTS_ROOT}/${COND_NAME}/checkpoint_last.pt"
        if [ ! -f "${CHECKPOINT}" ]; then
            echo "      [${COND_NAME}] WARNING: checkpoint not found -- skipping."
            continue
        fi

        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}_layer${LAYER_IDX}_len${EVAL_LEN}"
            REG_REL_OUT="${REG_REL_PROBE_RESULTS}/${PROBE_TAG}.json"

            if [ -f "${REG_REL_OUT}" ]; then
                echo "      [${PROBE_TAG}] Result exists -- skipping."
                continue
            fi

            echo "      [${PROBE_TAG}] Training relative position probe (regression) ..."

            "${PY}" "${REPO_DIR}/nopos_experiments/encdec_future_mask/run_relative_position_probe_regression.py" \
                --checkpoint            "${CHECKPOINT}" \
                --data                  "${DATABIN}" \
                --split                 valid \
                --train-split           train \
                --probe-layer           "${LAYER_IDX}" \
                --probe-updates         "${PROBE_MAX_UPDATES}" \
                --lr                    "${PROBE_LR}" \
                --max-tokens            "${PROBE_MAX_TOKENS}" \
                --pairs-per-seq         "${REL_PROBE_PAIRS_PER_SEQ}" \
                --eval-tokens-per-sample "${EVAL_LEN}" \
                --output                "${REG_REL_OUT}"

            echo "      [${PROBE_TAG}] Done."
        done
    done
done

# =============================================================================
# Step 7: Summarise results
# =============================================================================
CURRENT_STAGE="Summary"
echo ""
echo "======================================================"
echo "  POSITION PROBE RESULTS"
echo "======================================================"

for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
    echo ""
    echo "--- Eval sequence length: ${EVAL_LEN} ---"
    echo ""
    printf "%-25s %5s  %8s %8s  %8s %8s %8s  %8s %8s\n" \
        "CONDITION" "LAYER" "ABS_MAE" "ABS_ACC" "REL_MAE" "REL_ACC" "REL_DIR" "RREG_MAE" "RREG_DIR"
    printf "%-25s %5s  %8s %8s  %8s %8s %8s  %8s %8s\n" \
        "-------------------------" "-----" "--------" "--------" "--------" "--------" "--------" "--------" "--------"

    for cond_str in "${CONDITIONS[@]}"; do
        IFS='|' read -r COND_NAME NOPOS_FLAG ENCODER_MASK COND_EXTRA <<< "${cond_str}"

        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}_layer${LAYER_IDX}_len${EVAL_LEN}"

            ABS_OUT="${PROBE_RESULTS}/${PROBE_TAG}.json"
            REL_OUT="${REL_PROBE_RESULTS}/${PROBE_TAG}.json"

            if [ -f "${ABS_OUT}" ]; then
                ABS_MAE=$("${PY}" -c "import json; d=json.load(open('${ABS_OUT}')); print(f\"{d['probe_mae']:.3f}\")")
                ABS_ACC=$("${PY}" -c "import json; d=json.load(open('${ABS_OUT}')); print(f\"{d['probe_acc']:.3f}\")")
            else
                ABS_MAE="N/A"; ABS_ACC="N/A"
            fi

            if [ -f "${REL_OUT}" ]; then
                REL_MAE=$("${PY}" -c "import json; d=json.load(open('${REL_OUT}')); print(f\"{d['rel_probe_mae']:.3f}\")")
                REL_ACC=$("${PY}" -c "import json; d=json.load(open('${REL_OUT}')); print(f\"{d['rel_probe_acc']:.3f}\")")
                REL_DIR=$("${PY}" -c "import json; d=json.load(open('${REL_OUT}')); print(f\"{d['rel_probe_dir_acc']:.3f}\")")
            else
                REL_MAE="N/A"; REL_ACC="N/A"; REL_DIR="N/A"
            fi

            REG_REL_OUT="${REG_REL_PROBE_RESULTS}/${PROBE_TAG}.json"
            if [ -f "${REG_REL_OUT}" ]; then
                RREG_MAE=$("${PY}" -c "import json; d=json.load(open('${REG_REL_OUT}')); print(f\"{d['reg_rel_probe_mae']:.3f}\")")
                RREG_DIR=$("${PY}" -c "import json; d=json.load(open('${REG_REL_OUT}')); print(f\"{d['reg_rel_probe_dir_acc']:.3f}\")")
            else
                RREG_MAE="N/A"; RREG_DIR="N/A"
            fi

            printf "%-25s %5s  %8s %8s  %8s %8s %8s  %8s %8s\n" \
                "${COND_NAME}" "${LAYER_IDX}" "${ABS_MAE}" "${ABS_ACC}" "${REL_MAE}" "${REL_ACC}" "${REL_DIR}" "${RREG_MAE}" "${RREG_DIR}"
        done
    done
done

echo ""
echo "======================================================"
echo "  Probe experiment complete."
echo "  Absolute results           : ${PROBE_RESULTS}/"
echo "  Relative results (cls)     : ${REL_PROBE_RESULTS}/"
echo "  Relative results (reg)     : ${REG_REL_PROBE_RESULTS}/"
echo "======================================================"
