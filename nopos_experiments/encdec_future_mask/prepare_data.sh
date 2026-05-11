#!/bin/bash
# Download WikiText-2 (lightweight stand-in for WikiText-103) and binarize.
# WT-2 has the same format and dictionary structure as WT-103 but is ~70x smaller,
# making it tractable for a 100-update smoke trial on a single GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$ROOT"

DATA_RAW="wikitext-2"
DATA_BIN="data-bin/wikitext-2"

if [ ! -d "$DATA_BIN" ]; then
    if [ ! -d "$DATA_RAW" ]; then
        echo "Downloading WikiText-2..."
        if ! command -v wget >/dev/null; then
            echo "wget not found; please install or download wikitext-2-v1.zip manually." >&2
            exit 1
        fi
        wget --quiet https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
        unzip -q wikitext-2-v1.zip
        rm wikitext-2-v1.zip
    fi
    echo "Binarizing..."
    python3 "$ROOT/fairseq_cli/preprocess.py" \
        --only-source \
        --trainpref "$DATA_RAW/wiki.train.tokens" \
        --validpref "$DATA_RAW/wiki.valid.tokens" \
        --testpref  "$DATA_RAW/wiki.test.tokens" \
        --destdir   "$DATA_BIN" \
        --workers 4
fi
echo "Data ready at $DATA_BIN"
