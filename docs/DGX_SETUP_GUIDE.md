# ToxiFusionNet: NVIDIA DGX H100 Pod Execution & Reproducibility Guide

**Target Hardware:** NVIDIA DGX H100 (8x H100 80GB SXM5, PCIe Gen5, Ubuntu 22.04 LTS / CUDA 12.x)  
**Repository:** `https://github.com/ShubhamRaz/toxifusionnet.git`  
**Author:** Senior ML Data Engineer & Research Scientist  

---

## 1. Storage & Persistence Policy for Ephemeral Pods

> [!CAUTION]
> **Ephemeral Pod Notice:** Cloud compute pods (RunPod, Lambda, Brev, DGX Cloud) often store their root filesystem ephemerally. When a pod is stopped or deleted, local files are lost.
> Always link your runs directory to a persistent network volume (e.g. `/workspace`, `/mnt/storage`, or an S3/GCS bucket).

### Directory Configuration
- **Repository Location:** `/workspace/toxifusionnet`
- **Persistent Data & Artifact Store:** `/workspace/persistent_storage`
- **Scratch / Temp Directory:** `/tmp/scratch`

---

## 2. Step-by-Step DGX Setup Instructions

### Step 1: Clone or Pull the Published Commit

```bash
# Navigate to persistent volume
cd /workspace

# Clone the repository
git clone https://github.com/ShubhamRaz/toxifusionnet.git
cd toxifusionnet

# Checkout the published reproducible release branch
git checkout main
git pull origin main

# Verify commit hash
git rev-parse HEAD
```

---

### Step 2: Clean Environment Setup & Dependency Installation

```bash
# Create and activate Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.x support (optimized for H100 Hopper architecture)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify NVIDIA GPU recognition
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Count:', torch.cuda.device_count(), '| Device Name:', torch.cuda.get_device_name(0))"

# Install ToxiFusionNet in editable mode with development, explanation, and server dependencies
pip install -e ".[explain,serve,dev]"
```

---

### Step 3: Dataset Retrieval & Reconstruction

If raw datasets are stored in object storage or persistent volume:
```bash
# Option A: Reconstruct curated and cleaned datasets from verified raw source
python3 scripts/build_cleaned_pipeline.py

# Option B: Verify and download GloVe embeddings cache (for CPU baseline / comparison runs)
bash scripts/get_glove.sh

# Option C: If copying raw images from persistent archive
mkdir -p data/raw/bully_data
unzip -q /workspace/persistent_storage/bully_data.zip -d data/raw/bully_data/
```

---

### Step 4: Run Pre-Flight Validation & CV-Integrity Tests

Before launching GPU training, verify dataset checksums, label masks, and split isolation:

```bash
# 1. Run automated dataset integrity audit
python3 scripts/validate_cleaned_datasets.py

# 2. Run data integrity test suite
python3 -m pytest tests/test_data_integrity_audit.py -v

# 3. Run matched cross-validation integrity test suite
python3 -m pytest tests/test_matched_cv_integrity.py -v

# 4. Run full unit test suite (fast, offline, ~45s)
python3 -m pytest tests/ -k "not e2e_training"
```

---

### Step 5: Execute Pre-Flight Smoke Test (CPU/GPU Sanity Check)

Run a fast, 1-epoch smoke test to verify forward/backward passes and metric logging:

```bash
# Smoke test on synthetic pipeline
python3 -m toxifusionnet.cli train --config configs/smoke_test.yaml

# Smoke test on small MultiBully subset
python3 -m toxifusionnet.cli train --config configs/smoke_multibully.yaml
```

---

## 3. Canonical GPU Training & Evaluation Runs

### Experiment E1d: MultiBully Multi-Task Text (Class-Weighted)
Full 5-fold cross-validation on canonical MultiBully:

```bash
python3 -m toxifusionnet.cli cv \
    --config configs/exp1d_multibully_text_weighted.yaml \
    --data data/processed/multibully.csv \
    --folds-cache data/processed/multibully_v2_cv_folds.csv \
    --out-dir /workspace/persistent_storage/runs/e1d_multibully_text_weighted \
    --folds 5 \
    --seed 42
```

### Experiment E6b: MultiBully + HOT Gold Auxiliary Transfer (Outer-Test Isolated)
Full 5-fold cross-validation with auxiliary HOT text injected strictly into train partitions:

```bash
python3 -m toxifusionnet.cli cv \
    --config configs/exp6b_combined_text_gold_weighted.yaml \
    --data data/processed/multibully.csv \
    --aux-data data/processed/hot_gold_only.csv \
    --folds-cache data/processed/multibully_v2_cv_folds.csv \
    --out-dir /workspace/persistent_storage/runs/e6b_combined_text_gold_weighted \
    --folds 5 \
    --seed 42
```

### Multimodal Experiments (CLIP ViT-B/32 + Text Transformer)
```bash
# E3: Multimodal MultiBully
python3 -m toxifusionnet.cli cv \
    --config configs/exp3_multibully_clip.yaml \
    --data data/processed/multibully.csv \
    --folds-cache data/processed/multibully_v2_cv_folds.csv \
    --out-dir /workspace/persistent_storage/runs/e3_multibully_clip \
    --folds 5 \
    --seed 42
```

---

## 4. Post-Training Synchronization & Result Preservation

Before stopping or deleting the DGX pod, synchronize run artifacts to persistent storage:

```bash
# 1. Export summary metrics and prediction CSVs
mkdir -p /workspace/persistent_storage/published_results/$(date +%Y%m%d)_dgx_run

cp -r results/v3_journal_final /workspace/persistent_storage/published_results/$(date +%Y%m%d)_dgx_run/
cp data/manifests/*.json /workspace/persistent_storage/published_results/$(date +%Y%m%d)_dgx_run/

# 2. If using S3 / GCS, sync immediately
# aws s3 sync /workspace/persistent_storage/ s3://my-toxifusionnet-bucket/runs/ --exclude "*.tmp"
# rsync -avz /workspace/persistent_storage/ user@headnode:/shared/toxifusionnet_results/

echo "Synchronization complete. Pod may now be safely decommissioned."
```
