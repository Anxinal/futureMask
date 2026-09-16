#!/bin/bash
#SBATCH --job-name=attnrec
#SBATCH --output=attnrec_%j.out
#SBATCH --error=attnrec_%j.err
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-long

set -euo pipefail

# =============================================================================
# -- SELF-ATTENTION RECOVERY EXPERIMENT ---------------------------------------
#
# Stage A trains a teacher: one layer, one UNMASKED full-width head, MLM objective.
# Stage B freezes it and trains students whose heads are masked, to regress the
# teacher's attention sublayer output:
#
#   CF   one causal + one future-only head  -- the experimental arm
#   CC   two causal heads                   -- control, no access to the future
#   BB   two unmasked heads                 -- ceiling, bounds the non-mask error
#
# The result is the ORDERING of rel_l2 across the three, read against mae_baseline
# (the error of predicting a constant). See nopos_experiments/attn_recovery/PLAN.md.
# =============================================================================

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

# ---- Architecture (shared by teacher and students) --------------------------
EMBED_DIM=1024
FFN_DIM=4096
TOKENS_PER_SAMPLE=512

SEED=1924

# ---- Stage A: teacher ------------------------------------------------------
# One unmasked head, MLM objective. Not meant to be a strong LM -- a single layer
# with no positional embeddings cannot be -- only a fixed, well-defined attention
# function for the students to chase. What matters is that it trains at all: the
# loss is in bits and starts near log2(vocab) ~= 18 on WikiText-103.
TEACHER_MAX_UPDATES=20000
TEACHER_MAX_TOKENS=8192
TEACHER_LR=5e-4
TEACHER_WARMUP=4000
TEACHER_VALIDATE_EVERY=2000

# ---- Stage B: students ------------------------------------------------------
STUDENT_MAX_UPDATES=5000
STUDENT_MAX_TOKENS=8192
STUDENT_LR=1e-3
STUDENT_WARMUP=500
STUDENT_VALIDATE_EVERY=500

# Student embedding table. The student inherits nothing from the teacher: 'random'
# gives it its own table, freshly initialised and retrained from scratch, so every
# student parameter is learned. 'copy' would start it from the teacher's weights;
# 'shared' would reuse the teacher's tensor and freeze it.
# Note 'random'/'copy' add |vocab| x d trainable parameters (~274M on WikiText-103 at
# d=1024), which dominates the student's optimizer state and step time.
STUDENT_EMBED=${STUDENT_EMBED:-random}

# ---- Conditions: NAME|HEAD_MASK_SPEC ----------------------------------------
CONDITIONS=(
    "CF|C,F"
    "CC|C,C"
    "BB|B,B"
)

# ---- Masking (identical in both stages, so the students see the teacher's
#      training distribution) --------------------------------------------------
MASK_PROB=0.15
LEAVE_UNMASKED_PROB=0.1
RANDOM_TOKEN_PROB=0.1

# ---- Behaviour knobs --------------------------------------------------------
# FRESH_START : clear a run directory that has no completion marker. fairseq
#               auto-resumes from <save-dir>/checkpoint_last.pt, which across code
#               changes either aborts the job or silently restores stale weights.
# SKIP_SETUP  : 1 = use the python already on PATH instead of building a conda env.
#               Useful for running this locally against an existing install.
# SMOKE       : 1 = tiny model, ~100 updates -- wiring validation, not a result.
# RUN_VERIFY  : 1 = run verify_attn_recovery.py after stage A and abort on failure.
FRESH_START=${FRESH_START:-1}
SKIP_SETUP=${SKIP_SETUP:-0}
SMOKE=${SMOKE:-0}
RUN_VERIFY=${RUN_VERIFY:-1}

if [ "${SMOKE}" = "1" ]; then
    echo "### SMOKE MODE: tiny model, ~100 updates. Wiring check only. ###"
    EMBED_DIM=128
    FFN_DIM=512
    TOKENS_PER_SAMPLE=128
    TEACHER_MAX_UPDATES=100
    TEACHER_WARMUP=10
    TEACHER_VALIDATE_EVERY=50
    TEACHER_MAX_TOKENS=2048
    STUDENT_MAX_UPDATES=100
    STUDENT_WARMUP=10
    STUDENT_VALIDATE_EVERY=50
    STUDENT_MAX_TOKENS=2048
fi

# ---- Paths ------------------------------------------------------------------
# Under SLURM the script is copied to a temp dir, so $0's dirname is wrong --
# SLURM_SUBMIT_DIR is the directory sbatch ran in. Fall back to the script's own
# directory for local runs.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_RAW="${REPO_DIR}/wt103-raw/wikitext-103"
DATABIN="${REPO_DIR}/data-bin/wikitext-103"
CHECKPOINTS_ROOT="${REPO_DIR}/checkpoints_attn_recovery"
RESULTS_DIR="${REPO_DIR}/attn_recovery_results"
TEACHER_DIR="${CHECKPOINTS_ROOT}/teacher_seed${SEED}"
TEACHER_CKPT="${TEACHER_DIR}/checkpoint_last.pt"

if [ "${SMOKE}" = "1" ]; then
    CHECKPOINTS_ROOT="${REPO_DIR}/checkpoints_attn_recovery_smoke"
    RESULTS_DIR="${REPO_DIR}/attn_recovery_results_smoke"
    TEACHER_DIR="${CHECKPOINTS_ROOT}/teacher_seed${SEED}"
    TEACHER_CKPT="${TEACHER_DIR}/checkpoint_last.pt"
fi

echo "======================================================"
echo "  Self-Attention Recovery Experiment"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Node         : $(hostname)"
echo "  Conditions   : ${#CONDITIONS[@]} (${CONDITIONS[*]})"
echo "  Embed dim    : ${EMBED_DIM}   FFN: ${FFN_DIM}   T: ${TOKENS_PER_SAMPLE}"
echo "  Seed         : ${SEED}"
echo "  Repo         : ${REPO_DIR}"
echo "======================================================"

# =============================================================================
# -- CUDA / GPU diagnostics ---------------------------------------------------
# =============================================================================
CURRENT_STAGE="GPU / CUDA detection"
echo "--- GPU info ---"
GPU_OK=1
nvidia-smi || { echo "WARNING: nvidia-smi failed -- no GPU visible"; GPU_OK=0; }

CUDA_VER=""
if [ "${GPU_OK}" = "1" ]; then
    CUDA_VER=$(nvidia-smi 2>/dev/null \
        | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' \
        | head -1 \
        | tr -d '.') || CUDA_VER=""
fi
case "${CUDA_VER:-0}" in
    124|125|126) TORCH_CU="cu124" ;;
    121|122|123) TORCH_CU="cu121" ;;
    118|119|120) TORCH_CU="cu118" ;;
    *)           TORCH_CU="cu121"
                 echo "NOTE: could not detect CUDA version (got '${CUDA_VER}'), defaulting to cu121" ;;
esac
echo "PyTorch wheel: ${TORCH_CU}"

# Precision flags: fp16 needs a GPU. A CPU smoke run falls back to fp32.
if [ "${GPU_OK}" = "1" ]; then
    PRECISION_ARGS=(--fp16)
else
    PRECISION_ARGS=(--cpu)
    echo "NOTE: running on CPU in fp32."
fi
echo "---"

# =============================================================================
# Step 1: Python 3.10
# =============================================================================
CURRENT_STAGE="Step 1 -- Miniconda / Python 3.10 setup"

if [ "${SKIP_SETUP}" = "1" ]; then
    echo "[1/6] SKIP_SETUP=1 -- using the python already on PATH."
    PY="$(command -v python3 || command -v python)"
else
    MINICONDA_DIR="${HOME}/miniconda3"

    if [ ! -d "${MINICONDA_DIR}" ]; then
        echo "[1/6] Miniconda not found -- installing into ${MINICONDA_DIR} ..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O /tmp/miniconda_install.sh
        bash /tmp/miniconda_install.sh -b -p "${MINICONDA_DIR}"
        rm /tmp/miniconda_install.sh
    else
        echo "[1/6] Miniconda already present at ${MINICONDA_DIR}."
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
    # Pin pip as required by the fairseq Cython build
    "${PY}" -m pip install -q "pip<22.0.4"
fi
echo "      Python: $("${PY}" --version)  at ${PY}"

# =============================================================================
# Step 2: Install dependencies
# =============================================================================
CURRENT_STAGE="Step 2 -- dependency installation (torch / fairseq)"

if [ "${SKIP_SETUP}" = "1" ]; then
    echo "[2/6] SKIP_SETUP=1 -- assuming torch and fairseq are installed."
else
    echo "[2/6] Installing dependencies ..."
    "${PY}" -m pip install "torch>=2.5.0" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CU}"
    "${PY}" -m pip install -q numpy datasets

    # Install fairseq editable, bypassing pyproject.toml (same as colab.ipynb)
    cd "${REPO_DIR}"
    mv pyproject.toml pyproject.toml.bak
    "${PY}" -m pip install -q -e . --no-build-isolation
    mv pyproject.toml.bak pyproject.toml
fi

cd "${REPO_DIR}"
"${PY}" -c "
import torch, fairseq
print('torch    :', torch.__version__)
print('fairseq  :', fairseq.__version__)
print('CUDA ok  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU      :', torch.cuda.get_device_name(0))
"
# Fail early if the new model/criterion did not register.
"${PY}" -c "
from fairseq.models import ARCH_MODEL_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY
for name in ('simple_attn_teacher_base', 'simple_attn_student_base'):
    assert name in ARCH_MODEL_REGISTRY, f'arch {name} not registered'
assert 'attn_recovery' in CRITERION_REGISTRY, 'criterion attn_recovery not registered'
print('registry : simple_attn_* and attn_recovery OK')
"

# =============================================================================
# Step 3: Download & preprocess WikiText-103
# =============================================================================
CURRENT_STAGE="Step 3 -- WikiText-103 download / preprocessing"
echo "[3/6] Preparing WikiText-103 data ..."

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

mkdir -p "${CHECKPOINTS_ROOT}" "${RESULTS_DIR}"

# =============================================================================
# Step 4: Stage A -- train the unmasked teacher (MLM)
# =============================================================================
CURRENT_STAGE="Step 4 -- teacher (stage A)"

if [ -f "${TEACHER_CKPT}" ]; then
    echo "[4/6] Teacher checkpoint already present -- skipping."
    echo "      ${TEACHER_CKPT}"
else
    echo "[4/6] Training the unmasked teacher ..."
    # No checkpoint => whatever is in here is from a partial or older run, and the
    # students load this with strict=True.
    if [ "${FRESH_START}" = "1" ] && [ -d "${TEACHER_DIR}" ]; then
        echo "      Clearing stale ${TEACHER_DIR}"
        rm -rf "${TEACHER_DIR}"
    fi
    mkdir -p "${TEACHER_DIR}"

    TEACHER_ARGS=(
        "${DATABIN}"
        --task                          masked_lm
        --criterion                     masked_lm
        --arch                          simple_attn_teacher_base
        --head-mask-spec                B
        --encoder-embed-dim             "${EMBED_DIM}"
        --encoder-ffn-embed-dim         "${FFN_DIM}"
        --dropout                       0.1
        --sample-break-mode             none
        --tokens-per-sample             "${TOKENS_PER_SAMPLE}"
        --mask-prob                     "${MASK_PROB}"
        --leave-unmasked-prob           "${LEAVE_UNMASKED_PROB}"
        --random-token-prob             "${RANDOM_TOKEN_PROB}"
        --optimizer                     adam
        --adam-betas                    "(0.9, 0.98)"
        --weight-decay                  0.01
        --clip-norm                     1.0
        --lr                            "${TEACHER_LR}"
        --lr-scheduler                  inverse_sqrt
        --warmup-updates                "${TEACHER_WARMUP}"
        --max-tokens                    "${TEACHER_MAX_TOKENS}"
        --max-update                    "${TEACHER_MAX_UPDATES}"
        --validate-interval-updates     "${TEACHER_VALIDATE_EVERY}"
        --skip-invalid-size-inputs-valid-test
        "${PRECISION_ARGS[@]}"
        --save-dir                      "${TEACHER_DIR}"
        --save-interval-updates         "${TEACHER_MAX_UPDATES}"
        --keep-last-epochs              1
        --no-epoch-checkpoints
        --log-interval                  100
        --log-format                    json
        --num-workers                   4
        --seed                          "${SEED}"
    )

    printf '      %s\n' "python -m fairseq_cli.train ${TEACHER_ARGS[*]}"
    "${PY}" -m fairseq_cli.train "${TEACHER_ARGS[@]}" 2>&1 | tee "${TEACHER_DIR}/train.log"
    echo "      Teacher training done -- ${TEACHER_CKPT}"
fi

if [ ! -f "${TEACHER_CKPT}" ]; then
    echo "FATAL: teacher checkpoint missing at ${TEACHER_CKPT}" >&2
    exit 1
fi

# =============================================================================
# Step 5: Verify the setup before spending GPU hours on the students
# =============================================================================
CURRENT_STAGE="Step 5 -- verification"

if [ "${RUN_VERIFY}" = "1" ]; then
    echo "[5/6] Verifying masks, frozen teacher and optimisation ..."
    VERIFY_ARGS=(
        nopos_experiments/attn_recovery/verify_attn_recovery.py
        --teacher-checkpoint "${TEACHER_CKPT}"
        --data               "${DATABIN}"
        --student-embed      "${STUDENT_EMBED}"
    )
    [ "${GPU_OK}" = "1" ] || VERIFY_ARGS+=(--cpu)
    "${PY}" "${VERIFY_ARGS[@]}"
else
    echo "[5/6] RUN_VERIFY=0 -- skipping verification."
fi

# =============================================================================
# Step 6: Stage B -- train the masked students against the frozen teacher
# =============================================================================
CURRENT_STAGE="Step 6 -- students (stage B)"
echo "[6/6] Training students ..."

for cond_str in "${CONDITIONS[@]}"; do
    IFS='|' read -r COND_NAME SPEC <<< "${cond_str}"

    TAG="${COND_NAME}_seed${SEED}"
    SAVE_DIR="${CHECKPOINTS_ROOT}/${TAG}"
    RESULT_FILE="${RESULTS_DIR}/${TAG}.json"

    if [ -f "${RESULT_FILE}" ]; then
        echo "      [${TAG}] Result exists -- skipping."
        continue
    fi

    echo "      [${TAG}] spec='${SPEC}' -- training ..."
    # RESULT_FILE is the completion marker; without it, anything in SAVE_DIR is from
    # an incomplete or older run.
    if [ "${FRESH_START}" = "1" ] && [ -d "${SAVE_DIR}" ]; then
        rm -rf "${SAVE_DIR}"
    fi
    mkdir -p "${SAVE_DIR}"

    # --no-save: the student holds the frozen teacher as a submodule, so every
    # checkpoint would duplicate the (embedding-dominated) teacher. The metrics are
    # read from the JSON log instead.
    STUDENT_ARGS=(
        "${DATABIN}"
        --task                          masked_lm
        --criterion                     attn_recovery
        --arch                          simple_attn_student_base
        --teacher-checkpoint            "${TEACHER_CKPT}"
        --head-mask-spec                "${SPEC}"
        # Every student parameter trains, embeddings included. 'copy' starts them from
        # the teacher's table; 'shared' would reuse (and therefore freeze) it.
        --student-embed                 "${STUDENT_EMBED}"
        --dropout                       0.0
        --sample-break-mode             none
        --tokens-per-sample             "${TOKENS_PER_SAMPLE}"
        --mask-prob                     "${MASK_PROB}"
        --leave-unmasked-prob           "${LEAVE_UNMASKED_PROB}"
        --random-token-prob             "${RANDOM_TOKEN_PROB}"
        --optimizer                     adam
        --adam-betas                    "(0.9, 0.98)"
        --weight-decay                  0.0
        --clip-norm                     1.0
        --lr                            "${STUDENT_LR}"
        --lr-scheduler                  inverse_sqrt
        --warmup-updates                "${STUDENT_WARMUP}"
        --max-tokens                    "${STUDENT_MAX_TOKENS}"
        --max-update                    "${STUDENT_MAX_UPDATES}"
        --validate-interval-updates     "${STUDENT_VALIDATE_EVERY}"
        --skip-invalid-size-inputs-valid-test
        "${PRECISION_ARGS[@]}"
        --save-dir                      "${SAVE_DIR}"
        --no-save
        --log-interval                  50
        --log-format                    json
        --num-workers                   4
        --seed                          "${SEED}"
    )

    TRAIN_LOG="${SAVE_DIR}/train.log"
    printf '      %s\n' "python -m fairseq_cli.train ${STUDENT_ARGS[*]}"
    "${PY}" -m fairseq_cli.train "${STUDENT_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"

    # Recover the final validation metrics from the JSON log. --log-format json emits
    # one object per line, validation records prefixed "valid_" (progress_bar.py).
    RESULT_FILE="${RESULT_FILE}" TRAIN_LOG="${TRAIN_LOG}" COND="${COND_NAME}" \
    SPEC="${SPEC}" SEED="${SEED}" UPDATES="${STUDENT_MAX_UPDATES}" \
    "${PY}" - <<'PYEOF'
import json, os

log = os.environ["TRAIN_LOG"]
keys = ("valid_loss", "valid_mae", "valid_rel_l2", "valid_cos", "valid_mae_baseline")
found = {}
with open(log) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{") or "valid" not in line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for k in keys:
            if k in rec:
                found[k] = float(rec[k])
        # print() prefixes every key with the subset tag, so a validation record
        # carries "valid_num_updates"; only train_inner lines have it bare.
        for k in ("valid_num_updates", "num_updates"):
            if k in rec:
                try:
                    found["num_updates"] = int(float(rec[k]))
                except ValueError:
                    pass
                break

result = {
    "condition": os.environ["COND"],
    "head_mask_spec": os.environ["SPEC"],
    "seed": int(os.environ["SEED"]),
    "mae": found.get("valid_mae", float("nan")),
    "rel_l2": found.get("valid_rel_l2", float("nan")),
    "cos": found.get("valid_cos", float("nan")),
    "mae_baseline": found.get("valid_mae_baseline", float("nan")),
    "num_updates": found.get("num_updates", -1),
}
with open(os.environ["RESULT_FILE"], "w") as f:
    json.dump(result, f, indent=2)
print("      Saved", os.environ["RESULT_FILE"])
if not found:
    raise SystemExit("FATAL: no validation records found in " + log)
PYEOF

    echo "      [${TAG}] Done."
done

# =============================================================================
# Summary
# =============================================================================
CURRENT_STAGE="Summary"

RESULTS_DIR="${RESULTS_DIR}" SEED="${SEED}" CONDS="$(printf '%s\n' "${CONDITIONS[@]}")" \
"${PY}" - <<'PYEOF'
import json, math, os

results_dir = os.environ["RESULTS_DIR"]
seed = os.environ["SEED"]
conds = [c.split("|", 1) for c in os.environ["CONDS"].split("\n") if c.strip()]

print()
print("=" * 78)
print("  SELF-ATTENTION RECOVERY RESULTS")
print("=" * 78)
print()
print("  rel_l2 = ||student - teacher|| / ||teacher||  (lower = better recovery)")
print("  mae_base = error of predicting a constant; mae must beat it to mean anything")
print()
print(f"  {'COND':<6} {'SPEC':<8} {'MAE':>10} {'MAE_BASE':>10} {'REL_L2':>8} {'COS':>8} {'UPDATES':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

rows = {}
for name, spec in conds:
    path = os.path.join(results_dir, f"{name}_seed{seed}.json")
    if not os.path.exists(path):
        print(f"  {name:<6} {spec:<8} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        continue
    d = json.load(open(path))
    rows[name] = d
    print(
        f"  {name:<6} {spec:<8} {d['mae']:>10.5f} {d['mae_baseline']:>10.5f} "
        f"{d['rel_l2']:>8.4f} {d['cos']:>8.4f} {d['num_updates']:>8}"
    )

print()
if {"CF", "CC"} <= rows.keys():
    cf, cc = rows["CF"]["rel_l2"], rows["CC"]["rel_l2"]
    bb = rows.get("BB", {}).get("rel_l2", float("nan"))
    print(f"  CF vs CC : {cf:.4f} vs {cc:.4f}", end="")
    if not (math.isnan(cf) or math.isnan(cc)):
        if cc > 0:
            print(f"  ({100.0 * (cc - cf) / cc:+.1f}% error reduction from future access)")
        else:
            print()
    else:
        print()
    if not math.isnan(bb):
        print(f"  ceiling  : BB {bb:.4f} -- error not attributable to masking")
    print()
    print("  Read it as an ordering. BB <= CF < CC supports recovery; CF ~ CC means the")
    print("  causal+future split did not recover attention; all three alike means the")
    print("  target is too easy -- check mae against mae_base before concluding.")
print("=" * 78)
PYEOF

echo ""
echo "  Results : ${RESULTS_DIR}/"
echo "  Logs    : ${CHECKPOINTS_ROOT}/<cond>_seed${SEED}/train.log"
echo "======================================================"
