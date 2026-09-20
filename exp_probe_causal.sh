#!/bin/bash
#SBATCH --job-name=fixattn
#SBATCH --output=fixattn_%j.out
#SBATCH --error=fixattn_%j.err
#SBATCH --gpus=h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
# The probe reads a *pretrained* decoder, so we first train a plain NoPos LM whose
# Q/K projections are pinned to zero BEFORE the first optimizer step and held fixed
# throughout. Everything else -- V, out_proj, the FFN, the embeddings -- trains
# normally, so the network co-adapts to fixed uniform attention instead of learning
# content-based attention that is then thrown away at probe time.
#
# Expect a weak model: an LM restricted to a prefix mean of value vectors, with no
# positional embeddings, cannot do much. Loss is logged in bits and starts near
# log2(vocab) ~= 18 for WikiText-103; it should fall steadily but plateau well above
# a normally-attending LM. Not moving at all would be the bug.
#
# Uses `--arch fixed_attn_base_lm`, which shares fixed_attn_probe's arch config
# function so the two decoders cannot drift (the probe loads this with strict=True).
# This stage is the long pole of the job.
BASE_MAX_UPDATES=20000
BASE_MAX_TOKENS=8192
BASE_LR=5e-4

# ---- Stale-run handling -----------------------------------------------------
# fairseq auto-resumes from <save-dir>/checkpoint_last.pt. Across code changes
# that is a trap: a checkpoint written by an older criterion/architecture either
# aborts the job ("Criterion does not match") or, worse, silently restores stale
# weights. A run directory without its completion marker is by definition from an
# incomplete run, so clear it. Set to 0 to keep fairseq's resume-after-preemption
# behaviour instead.
FRESH_START=1

# ---- Probe training --------------------------------------------------------
PROBE_MAX_UPDATES=12000
PROBE_LR=1e-3
PROBE_WARMUP=500
PROBE_MAX_TOKENS=8192
PROBE_VALIDATE_EVERY=500

# ---- Probe target -----------------------------------------------------------
# position   : classify the absolute position t (criterion dpp_cross_entropy_fix).
# reciprocal : regress c/(t+1) + b (criterion reciprocal_position_probe). Under the
#              Q/K pin, 1/(t+1) is the prefix-mean weight -- the only
#              position-dependent quantity the network computes -- so this asks
#              whether the probe can read that signal directly, rather than invert it.
PROBE_TARGET=${PROBE_TARGET:-reciprocal}

# ---- Reciprocal target  c/(t+a)  --  ADJUST SCALE AND OFFSET HERE ------------
# Edit the defaults below, or override per job without touching this file:
#
#     RECIPROCAL_SCALE=512 RECIPROCAL_OFFSET=10 sbatch exp_probe_causal.sh
#
# RECIPROCAL_SCALE (c) : 0 = auto, T / sum_t 1/(t+a), which makes the mean target
#     exactly 1 (about 75 down to 0.15 at T=512, a=1). Unscaled, late targets are
#     ~2e-3, far from the probe's natural output range. Must be >= 0. Multiplicative,
#     so it cannot change r2, mad or accuracy -- only the loss's units.
# RECIPROCAL_OFFSET (a): denominator offset, default 1. Must be > 0. This one does
#     reshape the target. Measured at T=512:
#
#         a      range     R^2 of a probe resolving      ceiling for a LINEAR probe
#                          only t<10, guessing after
#         1       512x             0.959                          1.000
#         10       52x             0.613                          0.601
#         50       11x             0.277                          0.306
#
#     Larger a compresses the range and stops r2 flattering a probe that resolves
#     nothing late. The cost lands on the LINEAR arms only: the network computes
#     1/(t+1), and c/(t+a) is a Moebius transform of it that no linear map can
#     produce. The *_mlp arms (--non-linear-probe) can represent it and are uncapped.
#
#     a=1 is the default because it keeps the linear-vs-MLP contrast clean -- that
#     contrast is the experiment's point, and at a != 1 a linear arm's shortfall
#     mixes "not linearly decodable" with "target is not linear in the signal". At
#     a != 1 the script prints the linear ceiling; read the linear arms against it.
#
# There is deliberately no "+ b" outside the fraction: the probe head's own trainable
# bias absorbs such a term exactly (same weights, intercept shifted by b), so it
# cannot move any metric. Non-default values go into the run tag (e.g.
# "_recip_c512_a10"), so adjusting them starts fresh runs instead of being mistaken
# for finished ones and skipped.
RECIPROCAL_SCALE=${RECIPROCAL_SCALE:-0}
RECIPROCAL_OFFSET=${RECIPROCAL_OFFSET:-1}

# Read r2 TOGETHER WITH mad. At a=1 both the loss and r2 are dominated by early
# positions: a probe that resolves only t < 10 and guesses a constant afterwards
# scores r2 ~ 0.96 while being off by ~157 positions on average. mad and accuracy
# invert the prediction back to a position, and are comparable with the
# classification probe's numbers. Position-probe results keep their original,
# untagged names.

case "${PROBE_TARGET}" in
    position)
        PROBE_CRITERION=dpp_cross_entropy_fix
        TARGET_TAG=""
        ;;
    reciprocal)
        PROBE_CRITERION=reciprocal_position_probe
        TARGET_TAG="_recip"
        if [ "${RECIPROCAL_SCALE}" != "0" ]; then TARGET_TAG="${TARGET_TAG}_c${RECIPROCAL_SCALE}"; fi
        if [ "${RECIPROCAL_OFFSET}" != "1" ]; then TARGET_TAG="${TARGET_TAG}_a${RECIPROCAL_OFFSET}"; fi
        ;;
    *)
        echo "FATAL: PROBE_TARGET must be 'position' or 'reciprocal', got '${PROBE_TARGET}'" >&2
        exit 1
        ;;
esac

# ---- Conditions to run ------------------------------------------------------
# Each condition: NAME|EXTRA_FLAGS
#   NAME        : identifier for result dirs / tags
#   EXTRA_FLAGS : additional fairseq flags appended to the train command
#
# With Q/K pinned to zero every attention logit is 0, so attention is an exact
# uniform average over whatever the mask leaves visible:
#
# causal    = default causal mask. Position t reads the prefix mean
#             1/(t+1) * sum_{j<=t} v_j -- that 1/(t+1) is the only
#             position-dependent quantity in the network, and is the signal
#             being probed for.
# nocausal  = bidirectional mask. Every position reads the same global mean, so
#             hidden states vary only by token identity through the residual and
#             carry no position. This is the control and should sit at chance.
# *_mlp     = same masks, 2-layer MLP probe instead of linear. Bounds how much of
#             the signal is linearly decodable vs. merely present.
#
# All four arms share ONE causally-trained base LM; only the probe-time mask and
# probe type differ. Training a bidirectional base LM would be degenerate -- with
# next-token targets, position t would attend over token t+1, its own label.
CONDITIONS=(
    "causal|"
    "causal_mlp|--non-linear-probe"
    "nocausal|--decoder-head-mask-spec B"
    "nocausal_mlp|--decoder-head-mask-spec B --non-linear-probe"
)

# ---- Sequence lengths to evaluate ------------------------------------------
EVAL_LENGTHS=(512)

# ---- Probe layers (0 = embedding output, 1..N = decoder layer outputs) -----
# For a 2-layer decoder: 0 = embedding, 1 = layer 0, 2 = layer 1
PROBE_LAYERS=(0 1 2)

# ---- Resolved reciprocal target ---------------------------------------------
# Echo what the knobs above actually mean before anything trains. RECIPROCAL_SCALE=0
# is a sentinel for "auto", NOT a multiplier of zero: it resolves to the c that makes
# the mean of c/(t+a) equal 1.
if [ "${PROBE_TARGET}" = "reciprocal" ]; then
    if awk -v c="${RECIPROCAL_SCALE}" 'BEGIN{exit !(c < 0)}'; then
        echo "FATAL: RECIPROCAL_SCALE must be >= 0 (0 = auto), got '${RECIPROCAL_SCALE}'" >&2
        exit 1
    fi
    if awk -v a="${RECIPROCAL_OFFSET}" 'BEGIN{exit !(a <= 0)}'; then
        echo "FATAL: RECIPROCAL_OFFSET must be > 0, got '${RECIPROCAL_OFFSET}'" >&2
        exit 1
    fi

    for L in "${EVAL_LENGTHS[@]}"; do
        EFF_C=$(awk -v c="${RECIPROCAL_SCALE}" -v a="${RECIPROCAL_OFFSET}" -v T="${L}" \
            'BEGIN { if (c > 0) { printf "%.4f", c }
                     else { s = 0; for (t = 0; t < T; t++) s += 1 / (t + a); printf "%.4f", T / s } }')
        RANGE=$(awk -v c="${EFF_C}" -v a="${RECIPROCAL_OFFSET}" -v T="${L}" \
            'BEGIN { printf "%.4f .. %.4f", c / (T - 1 + a), c / a }')
        echo "Reciprocal target (T=${L}): ${EFF_C}/(t + ${RECIPROCAL_OFFSET})   [${RANGE}]"

        # What a perfect LINEAR probe can reach: the best least-squares fit of
        # c/(t+a) from the 1/(t+1) the network actually computes. Exactly 1 at a=1.
        if [ "${RECIPROCAL_OFFSET}" != "1" ]; then
            CEIL=$(awk -v c="${EFF_C}" -v a="${RECIPROCAL_OFFSET}" -v T="${L}" \
                'BEGIN { for (t = 0; t < T; t++) { u = 1/(t+1); y = c/(t+a)
                             su += u; sy += y; suu += u*u; suy += u*y; syy += y*y }
                         d = T*suu - su*su
                         w = (T*suy - su*sy) / d; b = (sy - w*su) / T
                         sse = syy - 2*w*suy - 2*b*sy + w*w*suu + 2*w*b*su + T*b*b
                         sst = syy - sy*sy/T
                         printf "%.3f", 1 - sse/sst }')
            echo "  a != 1: a perfect LINEAR probe is capped at R2 = ${CEIL} here;" \
                 "the *_mlp arms are not capped."
        fi
    done
    if [ "${RECIPROCAL_SCALE}" = "0" ]; then
        echo "  scale auto: c = T / sum_t 1/(t+a). Set RECIPROCAL_SCALE > 0 to override."
    fi
fi

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
    # No BASE_CKPT => whatever is in here is a partial or pre-Q/K-pin run. An old
    # checkpoint would resume cleanly (same criterion, same key set) and hand the
    # probe a base LM that never trained with the pin.
    if [ "${FRESH_START}" = "1" ] && [ -d "${BASE_LM_DIR}" ]; then
        echo "      Clearing stale ${BASE_LM_DIR}"
        rm -rf "${BASE_LM_DIR}"
    fi
    mkdir -p "${BASE_LM_DIR}"

    # fixed_attn_base_lm pins Q/K to zero before training and shares
    # fixed_attn_probe's arch config function, so the two decoders cannot drift --
    # the probe loads this checkpoint with strict=True. Do NOT swap in
    # --arch transformer_lm (or the bare model name fixed_attn_base): their
    # auto-registered arch functions are no-ops, so base_lm_architecture would not
    # run and the LM would be post-norm while the probe is pre-norm. That mismatch
    # loads cleanly under strict=True, hence the assert in build_model.
    # Do not add --share-decoder-input-output-embed either (changes the key set).
    #
    # --attention-dropout MUST stay 0.0: fairseq drops attention probabilities
    # without renormalising, which would break the exact uniform averaging.
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
        --log-interval                  100 \
        --log-format                    json \
        --num-workers                   0 \
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
            PROBE_TAG="${COND_NAME}${TARGET_TAG}_seed${SEED}_layer${LAYER_IDX}_len${EVAL_LEN}"
            SAVE_DIR="${CHECKPOINTS_ROOT}/${PROBE_TAG}"
            RESULT_FILE="${RESULTS_DIR}/${PROBE_TAG}.json"

            # Skip if result already exists
            if [ -f "${RESULT_FILE}" ]; then
                echo "      [${PROBE_TAG}] Result exists -- skipping."
                continue
            fi

            echo "      [${PROBE_TAG}] Training probe ..."
            # RESULT_FILE is the completion marker; it does not exist here, so any
            # checkpoint left in SAVE_DIR is from an incomplete or older run.
            if [ "${FRESH_START}" = "1" ] && [ -d "${SAVE_DIR}" ]; then
                echo "      Clearing stale ${SAVE_DIR}"
                rm -rf "${SAVE_DIR}"
            fi
            mkdir -p "${SAVE_DIR}"

            TRAIN_ARGS=(
                "${DATABIN}"
                --task                          language_modeling_position_probe
                --arch                          fixed_attn_probe
                --criterion                     "${PROBE_CRITERION}"
                --probe-target                  "${PROBE_TARGET}"
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
                --warmup-updates                "${PROBE_WARMUP}"
                --max-tokens                    "${PROBE_MAX_TOKENS}"
                --max-update                    "${PROBE_MAX_UPDATES}"
                --validate-interval-updates     "${PROBE_VALIDATE_EVERY}"
                --skip-invalid-size-inputs-valid-test
                --save-dir                      "${SAVE_DIR}"
                --save-interval-updates         "${PROBE_MAX_UPDATES}"
                --keep-last-epochs              1
                --no-epoch-checkpoints
                --log-interval                  50
                --log-format                    json
                --num-workers                   0
                --seed                          "${SEED}"
            )

            # Append condition-specific flags as individual array elements
            if [ -n "${COND_EXTRA}" ]; then
                # shellcheck disable=SC2206
                TRAIN_ARGS+=(${COND_EXTRA})
            fi
            # These belong to the reciprocal criterion's config, so passing them
            # alongside dpp_cross_entropy_fix would be rejected. The "=" form avoids
            # any value being parsed as an option name.
            #
            # The reciprocal probe trains in fp32. fp16 stores the probe's output with
            # a step of ~1/1024 of its magnitude, and neighbouring late positions
            # differ by only ~c/T^2: perfect predictions still invert exactly up to
            # about a=200, but the margin shrinks as larger a bunches the targets
            # together (98% accuracy at a=1000). fp32 keeps the offset knob safe across
            # its whole range on a model this small. The classification probe's logits
            # are not sensitive to this and keep fp16.
            if [ "${PROBE_TARGET}" = "reciprocal" ]; then
                TRAIN_ARGS+=(
                    "--reciprocal-scale=${RECIPROCAL_SCALE}"
                    "--reciprocal-offset=${RECIPROCAL_OFFSET}"
                )
            else
                TRAIN_ARGS+=(--fp16)
            fi

            TRAIN_LOG="${SAVE_DIR}/train.log"
            printf '      %s\n' "python -m fairseq_cli.train ${TRAIN_ARGS[*]}"
            # Tee so the summary step can recover accuracy / mad from the JSON log.
            "${PY}" -m fairseq_cli.train "${TRAIN_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"

            # Extract final validation metrics from the checkpoint + JSON log
            "${PY}" -c "
import json, os, sys

# Every metric in a row comes from the SAME (last) validation pass. val_loss used to
# be read from the checkpoint's extra_state['best'] -- the best loss over training --
# while accuracy/mad/r2 came from the last pass, so a row mixed two moments and
# disagreed whenever the probe wobbled or overfitted. best_val_loss is still reported
# alongside, clearly labelled.
#
# fairseq writes these records through the logging module, so each line reads
#   2026-09-17 10:00:00 | INFO | valid | {\"epoch\": 1, \"valid_mad\": ...}
# -- the JSON is a suffix, not the whole line. Matching on line.startswith('{')
# found nothing and silently recorded nan; parse from the first brace instead.
val_loss = accuracy = mad = r2 = float('nan')
num_updates = -1
try:
    with open('${TRAIN_LOG}') as f:
        for line in f:
            brace = line.find('{')
            if brace < 0:
                continue
            try:
                rec = json.loads(line[brace:])
            except ValueError:
                continue
            if 'valid_loss' in rec:
                val_loss = float(rec['valid_loss'])
            if 'valid_accuracy' in rec:
                accuracy = float(rec['valid_accuracy'])
            if 'valid_mad' in rec:
                mad = float(rec['valid_mad'])
            if 'valid_r2' in rec:
                r2 = float(rec['valid_r2'])
            if 'valid_num_updates' in rec:
                num_updates = int(float(rec['valid_num_updates']))
except OSError:
    pass

if val_loss != val_loss:
    print('FATAL: no validation records found in ${TRAIN_LOG}', file=sys.stderr)
    sys.exit(1)

# The checkpoint is optional now -- only the best-loss figure comes from it.
best_val_loss = float('nan')
ckpt_path = os.path.join('${SAVE_DIR}', 'checkpoint_last.pt')
if os.path.exists(ckpt_path):
    import torch
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    best_val_loss = state.get('extra_state', {}).get('best', float('nan'))

result = {
    'condition': '${COND_NAME}',
    'probe_target': '${PROBE_TARGET}',
    'probe_layer': ${LAYER_IDX},
    'eval_length': ${EVAL_LEN},
    'seed': ${SEED},
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,
    'r2': r2,
    'accuracy': accuracy,
    'mad': mad,
    'num_updates': num_updates,
}
# Record the target actually used, with the auto scale resolved to its number.
if '${PROBE_TARGET}' == 'reciprocal':
    T = ${EVAL_LEN}
    a = float('${RECIPROCAL_OFFSET}')
    result['reciprocal_scale'] = float('${RECIPROCAL_SCALE}') or T / sum(1.0 / (t + a) for t in range(T))
    result['reciprocal_offset'] = float('${RECIPROCAL_OFFSET}')

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
echo "  Probe target: ${PROBE_TARGET}"
if [ "${PROBE_TARGET}" = "reciprocal" ]; then
    echo "  Target c/(t+a) with c=${RECIPROCAL_SCALE} (0 = auto), a=${RECIPROCAL_OFFSET}."
    echo "  VAL_LOSS is the MSE in those units. Chance is R2 ~ 0."
    echo "  Read R2 together with MAD: R2 is dominated by early positions and stays"
    echo "  high for a probe with almost no late-position resolution."
else
    echo "  Loss is in bits. Chance for ${TOKENS_PER_SAMPLE} positions is"
    echo "  log2(${TOKENS_PER_SAMPLE} + 5) bits with accuracy ~0."
fi
echo "  Layer 0 (embedding, no positional embeddings) MUST sit at chance."

for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
    echo ""
    echo "--- Eval sequence length: ${EVAL_LEN} ---"
    echo ""
    printf "%-15s  %5s  %9s  %8s  %8s  %8s  %10s\n" "CONDITION" "LAYER" "VAL_LOSS" "R2" "ACC(%)" "MAD" "UPDATES"
    printf "%-15s  %5s  %9s  %8s  %8s  %8s  %10s\n" "---------------" "-----" "---------" "--------" "--------" "--------" "----------"

    for cond_str in "${CONDITIONS[@]}"; do
        IFS='|' read -r COND_NAME COND_EXTRA <<< "${cond_str}"

        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}${TARGET_TAG}_seed${SEED}_layer${LAYER_IDX}_len${EVAL_LEN}"
            RESULT_FILE="${RESULTS_DIR}/${PROBE_TAG}.json"

            if [ -f "${RESULT_FILE}" ]; then
                read -r VAL_LOSS R2 ACC MAD UPDATES <<< "$("${PY}" -c "
import json
d = json.load(open('${RESULT_FILE}'))
print(f\"{d['val_loss']:.4g}\",
      f\"{d.get('r2', float('nan')):.4f}\",
      f\"{d.get('accuracy', float('nan')):.3f}\",
      f\"{d.get('mad', float('nan')):.2f}\",
      d['num_updates'])
")"
            else
                VAL_LOSS="N/A"; R2="N/A"; ACC="N/A"; MAD="N/A"; UPDATES="N/A"
            fi

            printf "%-15s  %5s  %9s  %8s  %8s  %8s  %10s\n" \
                "${COND_NAME}" "${LAYER_IDX}" "${VAL_LOSS}" "${R2}" "${ACC}" "${MAD}" "${UPDATES}"
        done
    done
done

echo ""
echo "======================================================"
echo "  Fixed-attention probe experiment complete."
echo "  Results : ${RESULTS_DIR}/"
echo "======================================================"
