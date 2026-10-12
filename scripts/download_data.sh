#!/usr/bin/env bash
# Download the Kaggle copy of MultiBully (needs `pip install kaggle` and an API token in ~/.kaggle/kaggle.json).
# The Kaggle archive may also contain the meme IMAGES (large, unused by this text-only model): you only need the
# annotation sheet (.xlsx/.csv). Official source of the sheet: README of https://github.com/Jhaprince/MultiBully
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d pranee5/multi-bully -p data/raw --unzip
echo "--- files found:"; find data/raw -maxdepth 3 \( -name "*.xlsx" -o -name "*.csv" -o -name "*.xls" \) -print
echo "Next:  make inspect RAW=<annotation file>   then   make final RAW=<annotation file>"
