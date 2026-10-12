#!/usr/bin/env bash
# Pre-trained GloVe-Twitter-100 word vectors (Pennington et al. 2014; 2B tweets, 1.2M words; ~406 MB) as published in
# gensim-data. Used by configs/multibully_cpu.yaml to give the small from-scratch encoder a pre-trained lexical prior
# when no GPU / pre-trained transformer is available.
set -euo pipefail
mkdir -p data/external
curl -L --fail -o data/external/glove-twitter-100.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/glove-twitter-100.gz
ls -la data/external/glove-twitter-100.gz
