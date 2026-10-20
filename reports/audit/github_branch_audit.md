# GitHub Branch & Deliverable Audit

---

## 1. Remote & Local Repository Specification

- **GitHub Repository URL:** `https://github.com/ShubhamRaz/toxifusionnet`
- **Target Branch Inspected:** `feature/hinglish-balanced-dataset`
- **Audit Execution Date:** October 20, 2026
- **Auditor Role:** Senior ML Research Auditor & NLP Software Engineer

---

## 2. Git State & Remote Synchronization

```text
Local Working Tree: c:\Users\SHUBHAM RAJ\Desktop\Others\Toxic\toxifusionnet\toxifusionnet
Current Local Branch: feature/hinglish-balanced-dataset
Current Local HEAD SHA: 371a09e13f412ba771bb5e7144e311f67823f66c
Working Tree Status: Clean (0 uncommitted changes, 0 unstaged changes)
```

### Remote Tracking Status
A direct `git fetch origin` was performed. Inspection of remote branches reveals:
```text
  origin/HEAD -> origin/main (b5192e0)
  origin/feature/simplified-main-categories (7a82886)
  origin/main (b5192e0)
  origin/publish/curated-dataset-dgx (b5192e0)
  origin/vivek (3ed5fff)
```

### Critical Finding: Branch Divergence / Remote Status
- **Finding:** The branch `feature/hinglish-balanced-dataset` **does not exist on the remote GitHub repository**.
- **Explanation:** It is currently a strictly local branch created off commit `33d18d9` (post-PR #2 merge `b5192e0`).
- **Commits present locally on `feature/hinglish-balanced-dataset` but absent from `origin`:**
  1. `b8ce353`: *feat: switch to balanced 48k Hinglish toxicity dataset and cleanup legacy data*
  2. `5d5927c`: *feat: merge MultiBully and HOT into combined_multibully_hot.csv (8,907 rows)*
  3. `71f6ea1`: *feat: merge all 3 datasets (MultiBully + HOT + Hinglish Balanced, 56,907 rows) and add config*
  4. `8cafe6f`: *clean: keep only Stage 3 tri-dataset, final model, and comprehensive report; remove all intermediate artifacts*
  5. `5965eff`: *docs: update master README with Stage 3 architecture, benchmarks, and inference guide*
  6. `11bfb3d`: *feat: configure run.ps1 and run_supercomputer.sh for 1-click training from scratch*
  7. `da4994d`: *clean: replace outdated reports with Stage 3 tri-dataset evaluation reports*
  8. `b9ed1f7`: *docs: add comprehensive unabridged ToxiFusionNet full project report*
  9. `56530c1`: *docs: add dataset provenance and references including Kaggle Hinglish Toxic Hate dataset*
  10. `371a09e`: *docs: add exact URLs for MultiBully (Kaggle) and HOT (GitHub) dataset references*

---

## 3. Tracked vs. Untracked Artifact Audit

| Artifact Path | Size | Tracked in Git? | Ignored by `.gitignore`? | Audit Status |
| :--- | :---: | :---: | :---: | :--- |
| `data/processed/combined_all_3_datasets.csv` | 15.6 MB | **YES** | No (`!data/processed/*.csv`) | Committed in `71f6ea1` |
| `data/processed/combined_all_3_datasets.meta.json` | 1.1 KB | **YES** | No | Committed in `71f6ea1` |
| `configs/combined_all_3_datasets.yaml` | 1.2 KB | **YES** | No | Committed in `71f6ea1` |
| `runs/exp_tri_dataset_xlmr/model.pt` | 1.13 GB | **NO** | **YES (Line 5: `runs/`)** | Local Only |
| `runs/exp_tri_dataset_xlmr/metrics.json` | 72.5 KB | **NO** | **YES (Line 5: `runs/`)** | Local Only |
| `runs/exp_tri_dataset_xlmr/preds_test.csv` | 1.81 MB | **NO** | **YES (Line 5: `runs/`)** | Local Only |
| `runs/exp_tri_dataset_xlmr/history.csv` | 2.9 KB | **NO** | **YES (Line 5: `runs/`)** | Local Only |
| `reports/TOXIFUSIONNET_FULL_PROJECT_REPORT.md`| 24.1 KB | **YES** | No | Committed in `b9ed1f7` |
| `reports/STAGE_3_FINAL_REPORT.md` | 7.6 KB | **YES** | No | Committed in `da4994d` |
| `reports/toxifusionnet_final_tri_dataset_evaluation.txt` | 2.5 KB | **YES** | No | Committed in `da4994d` |
| `README.md` | 8.8 KB | **YES** | No | Committed in `5965eff` |
| `run.ps1` | 2.0 KB | **YES** | No | Committed in `11bfb3d` |

### Key Reproducibility Implication
- Because `runs/` is gitignored, pushing `feature/hinglish-balanced-dataset` will push the dataset, configs, training scripts, and reports, but **not the binary weights checkpoint (`model.pt`) or prediction CSVs**.
- An external user cloning the repository from GitHub cannot run inference immediately using `python -m toxifusionnet.cli predict --run-dir runs/exp_tri_dataset_xlmr` until they either:
  1. Train from scratch using `python -m toxifusionnet.cli train --config configs/combined_all_3_datasets.yaml` (which works out of the box because `combined_all_3_datasets.csv` is tracked), OR
  2. Download `model.pt` from an external release asset (e.g., GitHub Releases or Hugging Face Hub).
