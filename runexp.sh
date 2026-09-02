#!/bin/bash
#SBATCH --job-name=nopos
#SBATCH --output=nopos_%j.out
#SBATCH --error=nopos_%j.err
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-long

set -euo pipefail

# =============================================================================
# ── ERROR TRAPPING ────────────────────────────────────────────────────────────
# =============================================================================
CURRENT_STAGE="initialisation"

error_handler() {
    local exit_code=$?
    local line_no=$1
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  FATAL ERROR — job aborted"
    echo "  Stage     : ${CURRENT_STAGE}"
    echo "  Line      : ${line_no}"
    echo "  Exit code : ${exit_code}"
    echo "  Command   : ${BASH_COMMAND}"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
}

trap 'error_handler ${LINENO}' ERR

# =============================================================================
# ── USER CONFIGURATION ────────────────────────────────────────────────────────
# =============================================================================

COND="CCCCFFFFdim1024-sin-1924"
SPEC="C,C,C,C,F,F,F,F"

MAX_TOKENS=32768
UPDATE_FREQ=2
MAX_UPDATES=52000
VALIDATE_EVERY=1000

# SLURM_SUBMIT_DIR is the directory where you ran sbatch — i.e. the repo root.
# Do NOT use $(dirname "$0"): SLURM copies the script to a temp dir first.
REPO_DIR="${SLURM_SUBMIT_DIR}"
DATA_RAW="${REPO_DIR}/wt103-raw/wikitext-103"
DATABIN="${REPO_DIR}/data-bin/wikitext-103"

echo "======================================================"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Node       : $(hostname)"
echo "  Condition  : ${COND}"
echo "  Head mask  : ${SPEC}"
echo "  Repo       : ${REPO_DIR}"
echo "======================================================"

# =============================================================================
# ── CUDA / GPU diagnostics ────────────────────────────────────────────────────
# =============================================================================
CURRENT_STAGE="GPU / CUDA detection"
echo "--- GPU info ---"
# nvidia-smi reads the kernel driver directly — no module load needed.
nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

# Detect driver-reported CUDA version to pick the matching PyTorch wheel.
# PyTorch pip wheels bundle their own CUDA runtime (libcudart), so no
# 'module load cuda' is needed — only the kernel driver (libcuda.so) must
# be present, which SLURM guarantees when --gpus / --gres=gpu is requested.
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
CURRENT_STAGE="Step 1 — Miniconda / Python 3.10 setup"

MINICONDA_DIR="${HOME}/miniconda3"

if [ ! -d "${MINICONDA_DIR}" ]; then
    echo "[1/4] Miniconda not found — installing into ${MINICONDA_DIR} ..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda_install.sh
    bash /tmp/miniconda_install.sh -b -p "${MINICONDA_DIR}"
    rm /tmp/miniconda_install.sh
else
    echo "[1/4] Miniconda already present at ${MINICONDA_DIR}."
fi

# Activate conda in this non-interactive shell
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

CONDA_ENV_NAME="nopos"
if conda env list | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
    echo "      Conda env '${CONDA_ENV_NAME}' already exists — skipping creation."
else
    echo "      Creating conda env '${CONDA_ENV_NAME}' with Python 3.10 ..."
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y -c conda-forge --override-channels
fi

conda activate "${CONDA_ENV_NAME}"
PY="$(which python)"
echo "      Python: $("${PY}" --version)  at ${PY}"

# Pin pip as required by the fairseq Cython build
"${PY}" -m pip install -q "pip<22.0.4"

# =============================================================================
# Step 2: Install dependencies
# =============================================================================
CURRENT_STAGE="Step 2 — dependency installation (torch / fairseq)"
echo "[2/4] Installing dependencies ..."

# --force-reinstall ensures a stale CPU-only torch from a previous failed run
# is replaced with the correct CUDA wheel, rather than pip skipping it.
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
    raise RuntimeError('CUDA not available — driver visible to nvidia-smi but not to PyTorch. CUDA version detected: ${CUDA_VER}, wheel: ${TORCH_CU}')
"

"${PY}" -m pip install -q numpy datasets

# Install fairseq editable, bypassing pyproject.toml (same as colab.ipynb)
cd "${REPO_DIR}"
mv pyproject.toml pyproject.toml.bak
"${PY}" -m pip install -q -e . --no-build-isolation
mv pyproject.toml.bak pyproject.toml

"${PY}" -c "import fairseq; print('fairseq OK:', fairseq.__version__)"

# =============================================================================
# Step 3: Download & preprocess WikiText-103
# =============================================================================
CURRENT_STAGE="Step 3 — WikiText-103 download / preprocessing"
echo "[3/4] Preparing WikiText-103 data ..."

if [ -d "${DATABIN}" ] && [ -n "$(ls -A "${DATABIN}" 2>/dev/null)" ]; then
    echo "      Preprocessed data already present — skipping."
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
# ── Step 4: Train ─────────────────────────────────────────────────────────────
# All output goes to nopos_<jobid>.out (no separate log file, no checkpoints).
# =============================================================================
CURRENT_STAGE="Step 4 — fairseq training"
echo "[4/4] Starting training (condition=${COND}, spec=${SPEC}}) ..."

TRAIN_ARGS=(
    "${DATABIN}"
    --task                     encoder_decoder_language_modeling
    --sample-break-mode        none
    --tokens-per-sample        512
    --encoder-prefix-fraction  0.5
    --arch                     transformer
    --encoder-layers           8   --decoder-layers          8
    --encoder-attention-heads  8   --decoder-attention-heads 8
    --encoder-embed-dim        1024 --decoder-embed-dim      1024
    --encoder-ffn-embed-dim    4096 --decoder-ffn-embed-dim  4096
    --share-all-embeddings
    --encoder-head-mask-spec   "${SPEC}"
    --dropout                  0.1  --attention-dropout      0.1
    --optimizer                adam --adam-betas "(0.9, 0.98)"
    --weight-decay             0.01 --clip-norm 1.0
    --lr                       7e-5 --lr-scheduler inverse_sqrt
    --warmup-updates           4000
    --criterion                cross_entropy
    --max-tokens               "${MAX_TOKENS}"
    --update-freq              "${UPDATE_FREQ}"
    --max-update               "${MAX_UPDATES}"
    --skip-invalid-size-inputs-valid-test
    --required-batch-size-multiple 1
    --validate-interval-updates "${VALIDATE_EVERY}"
    --fp16
    --no-save
    --log-interval             100
    --log-format               json
    --num-workers              4
    --seed                     1924
)

# Echo the resolved command so the seed is auditable in the .out file
printf '  %s\n' "python -m fairseq_cli.train ${TRAIN_ARGS[*]}"

"${PY}" -m fairseq_cli.train "${TRAIN_ARGS[@]}"

echo "======================================================"
echo "  Training complete."
echo "======================================================"

