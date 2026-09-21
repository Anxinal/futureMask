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

# ---- Probe loss: cross-entropy + distance penalty ---------------------------
# The probe classifies the absolute position (criterion dpp_cross_entropy_fix). Every
# token contributes one -log p(true position) term, so every position counts alike --
# unlike the reciprocal regression this replaced, whose 100x target range let the
# first ten positions carry 75% of the loss.
#
# Cross-entropy is blind to ORDER, though: predicting 301 for 300 costs the same as
# predicting 0. DISTANCE_PENALTY (lambda) adds the expected position distance,
#
#     loss = CE + lambda * E_p[|k - t|] / T
#
# whose gradient pushes each class down in proportion to how much farther it is than
# the current expected distance. Absolute distance, so no position is favoured.
# Override per job:   DISTANCE_PENALTY=20 sbatch exp_probe_causal.sh
#
# What to expect (SYNTHETIC pilot, 1-D signal at this model's information floor): the
# penalty lowers argmax MAD only modestly -- 13.4 -> 12.4 at lambda=100, for ~0.2
# points of accuracy -- because a classifier on a monotone signal already confuses
# mainly NEIGHBOURING positions. The median readout (mad_median, always reported)
# reached 12.1 with no penalty at all. On real, entangled hidden states the penalty
# may matter more; comparing lambda=0 against lambda=20..100 is the test.
#
# lambda goes into the run tag ("_dp0", "_dp20"), so sweeps never collide, and the
# old untagged classification results (earlier probe, parser-nan metrics) can never
# be mistaken for finished runs and skipped.
DISTANCE_PENALTY=${DISTANCE_PENALTY:-0}

if awk -v l="${DISTANCE_PENALTY}" 'BEGIN{exit !(l < 0)}'; then
    echo "FATAL: DISTANCE_PENALTY must be >= 0, got '${DISTANCE_PENALTY}'" >&2
    exit 1
fi
RUN_TAG="_dp${DISTANCE_PENALTY}"
echo "Probe loss: cross-entropy + ${DISTANCE_PENALTY} * expected position distance"

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
            PROBE_TAG="${COND_NAME}${RUN_TAG}_seed${SEED}_layer${LAYER_IDX}_len${EVAL_LEN}"
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
                --criterion                     dpp_cross_entropy_fix
                "--distance-penalty=${DISTANCE_PENALTY}"
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
                --fp16
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

            TRAIN_LOG="${SAVE_DIR}/train.log"
            printf '      %s\n' "python -m fairseq_cli.train ${TRAIN_ARGS[*]}"
            # Tee so the summary step can recover accuracy / mad from the JSON log.
            "${PY}" -m fairseq_cli.train "${TRAIN_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"

            # Extract final validation metrics from the checkpoint + JSON log
            "${PY}" -c "
import json, os, sys

# Every metric in a row comes from the SAME (last) validation pass. val_loss used to
# be read from the checkpoint's extra_state['best'] -- the best loss over training --
# while accuracy/mad came from the last pass, so a row mixed two moments and
# disagreed whenever the probe wobbled or overfitted. best_val_loss is still reported
# alongside, clearly labelled.
#
# fairseq writes these records through the logging module, so each line reads
#   2026-09-17 10:00:00 | INFO | valid | {\"epoch\": 1, \"valid_mad\": ...}
# -- the JSON is a suffix, not the whole line. Matching on line.startswith('{')
# found nothing and silently recorded nan; parse from the first brace instead.
val_loss = ce = exp_dist = accuracy = mad = mad_median = float('nan')
mad_q = [float('nan')] * 4
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
            if 'valid_ce' in rec:
                ce = float(rec['valid_ce'])
            if 'valid_exp_dist' in rec:
                exp_dist = float(rec['valid_exp_dist'])
            if 'valid_mad_median' in rec:
                mad_median = float(rec['valid_mad_median'])
            for b in range(4):
                key = 'valid_mad_q' + str(b + 1)
                if key in rec:
                    mad_q[b] = float(rec[key])
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
    'distance_penalty': float('${DISTANCE_PENALTY}'),
    'probe_layer': ${LAYER_IDX},
    'eval_length': ${EVAL_LEN},
    'seed': ${SEED},
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,
    'ce_bits': ce,
    'exp_dist': exp_dist,
    'accuracy': accuracy,
    'mad': mad,
    'mad_median': mad_median,
    'mad_by_quarter': mad_q,
    'num_updates': num_updates,
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
echo "  Loss: cross-entropy + ${DISTANCE_PENALTY} * expected position distance."
echo "  CE is pure cross-entropy in bits; chance for ${TOKENS_PER_SAMPLE} positions is"
echo "  log2(${TOKENS_PER_SAMPLE} + 5) ~ 9.0 bits, accuracy ~0.2%, MAD ~171."
echo "  MAD reads the argmax, MAD_MED the median of the predicted distribution;"
echo "  MAD_Q1..Q4 are argmax MAD within each quarter of the sequence."
echo "  Layer 0 (embedding, no positional embeddings) MUST sit at chance."

for EVAL_LEN in "${EVAL_LENGTHS[@]}"; do
    echo ""
    echo "--- Eval sequence length: ${EVAL_LEN} ---"
    echo ""
    printf "%-15s %5s %8s %7s %7s %8s %6s %6s %6s %6s %8s\n" \
        "CONDITION" "LAYER" "CE(bits)" "ACC(%)" "MAD" "MAD_MED" "MAD_Q1" "MAD_Q2" "MAD_Q3" "MAD_Q4" "UPDATES"
    printf "%-15s %5s %8s %7s %7s %8s %6s %6s %6s %6s %8s\n" \
        "---------------" "-----" "--------" "-------" "-------" "--------" "------" "------" "------" "------" "--------"

    for cond_str in "${CONDITIONS[@]}"; do
        IFS='|' read -r COND_NAME COND_EXTRA <<< "${cond_str}"

        for LAYER_IDX in "${PROBE_LAYERS[@]}"; do
            PROBE_TAG="${COND_NAME}${RUN_TAG}_seed${SEED}_layer${LAYER_IDX}_len${EVAL_LEN}"
            RESULT_FILE="${RESULTS_DIR}/${PROBE_TAG}.json"

            if [ -f "${RESULT_FILE}" ]; then
                read -r CE ACC MAD MADMED Q1 Q2 Q3 Q4 UPDATES <<< "$("${PY}" -c "
import json
d = json.load(open('${RESULT_FILE}'))
q = d.get('mad_by_quarter', [float('nan')] * 4)
print(f\"{d.get('ce_bits', float('nan')):.3f}\",
      f\"{d.get('accuracy', float('nan')):.3f}\",
      f\"{d.get('mad', float('nan')):.2f}\",
      f\"{d.get('mad_median', float('nan')):.2f}\",
      *[f\"{v:.1f}\" for v in q],
      d['num_updates'])
")"
            else
                CE="N/A"; ACC="N/A"; MAD="N/A"; MADMED="N/A"
                Q1="N/A"; Q2="N/A"; Q3="N/A"; Q4="N/A"; UPDATES="N/A"
            fi

            printf "%-15s %5s %8s %7s %7s %8s %6s %6s %6s %6s %8s\n" \
                "${COND_NAME}" "${LAYER_IDX}" "${CE}" "${ACC}" "${MAD}" "${MADMED}" \
                "${Q1}" "${Q2}" "${Q3}" "${Q4}" "${UPDATES}"
        done
    done
done

echo ""
echo "======================================================"
echo "  Fixed-attention probe experiment complete."
echo "  Results : ${RESULTS_DIR}/"
echo "======================================================"
