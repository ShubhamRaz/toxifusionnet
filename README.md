# ToxiFusionNet

**Sarcasm-aware multi-task transformer for hate-speech detection in Hindi-English code-mixed text.**
Detects hate speech *including sarcastic hate*, and for every hate post predicts its **target** and **severity**,
with token-level **rationales**. Studied here on the **MultiBully** meme-text dataset.

```
Input -> Preprocessing -> Sarcasm detection -> Hate detection (sarcasm-aware) -+-> Target
         (LID, normalise,  (cues + attention     (gated sarcasm fusion)         +-> Severity (low / moderate / high)
          transliterate)    pooling)                                           -> Output + rationales
```

## Results on MultiBully (real data; small GloVe-initialised encoder, CPU)
5,718 posts after cleaning (4,002 / 858 / 858 split). **Not a pre-trained transformer** -- see "What to do next".

| System (pooled 5-fold CV, 5,718 posts) | Hate macro-F1 | Hate AUROC | AUROC on sarcastic posts | Sarcasm macro-F1 | Target macro-F1 | Severity macro-F1 |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.671 | 0.763 | 0.775 | 0.559 | 0.374 | 0.372 |
| ToxiFusionNet without fusion | 0.652 | 0.761 | 0.760 | 0.560 | 0.371 | 0.404 |
| **ToxiFusionNet (full)** | 0.667 | 0.761 | 0.742 | 0.558 | 0.375 | 0.390 |

Test split, 3 seeds: hate macro-F1 -- hate-only transformer 0.676, multi-task without sarcasm 0.673, sarcasm without fusion 0.659, full 0.661 (seed sd about 0.01-0.02); TF-IDF 0.649.

**Honest reading.**
* The full model and TF-IDF are tied on hate detection; sarcasm-aware fusion did **not** improve hate detection, and *within sarcastic posts* the fusion model ranks hate **worse** (AUROC 0.742 vs 0.760 without fusion). Its lower miss rate on sarcastic hate is mostly a more liberal decision threshold.
* Sarcasm cannot be read well from the text alone here (AUROC about 0.58 for every system); my untested guess is that much of it is in the meme image.
* Target (society 68.0, organization 81.0 posts) and severity ("high": 30.0 posts) are barely above chance for the rare classes.
* "Fake-narrative" framing of a post drops hate macro-F1 from 0.643 to 0.484 (robustness test).
* Everything is in `results/RESULTS.md`; the report (`docs/ToxiFusionNet_Project_Report.docx`) discusses it, Section 10.

## Data decisions (MultiBully, `configs/schema_multibully.yaml`)
| Task | Column | Mapping |
|---|---|---|
| text | `Img-Text` | OCR text of the meme (images unused) |
| hate | `Text-Label` | Bully -> hate (text-level label; `Img-Text-Label` is the meme-level alternative, see `configs/schema_multibully_memelabel.yaml`) |
| sarcasm | `Sarcasm` | Yes / No (gold) |
| severity | `Harmful-Score` | Harmless -> low, Partially-Harmful -> moderate, Very-Harmful -> high (only 30 "high" posts) |
| target | `Target` | Individual / Organization / Community / Society. **The flow chart's caste / religion / gender / demography are not annotated**, so the dataset's taxonomy is used (`labels.target`; the other taxonomy is supported) |

5,865 sheet rows -> 5,854 labelled -> -4 flagged -> -132 duplicate texts -> **5,718 posts**, split 70/15/15 (+ 5-fold CV).
The corpus is mostly English with a Hinglish minority (heuristic: 38% of posts contain a Hindi token).
**The dataset itself is not included** (no licence stated); put the annotation csv/xlsx in `data/raw/`.

## What to do next
The results above use a **small from-scratch encoder with GloVe-Twitter vectors**, because the build machine had one CPU core
and no access to pre-trained transformers. The question that matters for the paper is whether **MuRIL / XLM-R / HingBERT**
change the picture:
```bash
make final RAW=data/raw/multibully.csv ENC=google/muril-base-cased          # GPU recommended, ~1-2 h (estimate)
make final RAW=data/raw/multibully.csv ENC=xlm-roberta-base
CV=1 make final ...                                                          # add pooled 5-fold CV
```
or use `notebooks/run_on_colab.ipynb` (not executed by the author). Everything is re-filled into `results/RESULTS.md`.

## Install & verify
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[explain,serve,dev]"
make test                 # 48 tests, offline, ~10 s
toxifusion predict --run-dir models/multibully_cpu_full_seed42 --text "Wah bhai wah, kya genius idea hai lol"   # shipped small model
```

## Reproduce the CPU study (what the report used)
```bash
bash scripts/get_glove.sh                    # 406 MB GloVe-Twitter-100 (cached subset is reused afterwards)
CFG=configs/multibully_cpu.yaml CV=1 bash scripts/run_final.sh data/raw/multibully.csv
```
Runs are resumable (finished run directories with identical configs are re-used).
Single steps: `toxifusion {inspect,prepare,describe,baselines,train,ablate,cv,robust,explain,bench,figures,report,predict,serve,compare}`;
any config value can be overridden, e.g. `-o model.use_fusion=false`.

## Design points
* **Multi-task with missing labels** (`-100` ignored per task); target / severity trained on hate posts only.
* **Sarcasm-aware fusion** `f = LayerNorm(h + g * tanh(W s))`, `g = sigmoid(W_g [h; s; P(sarcasm)])`.
* **Rationale supervision** (Mamun et al., 2024) is implemented but off for MultiBully (no rationale spans).
* **Evaluation:** macro-F1 / balanced accuracy per task, AUROC (overall and within sarcastic posts), false-negative rate and
  operating points, paired bootstrap (note: resamples test posts only, not training seeds), pooled cross-validation,
  robustness (spelling, transliteration, code-switch, emoji, fake-narrative framing), faithfulness, int8 latency.

## Layout
```
configs/    default · multibully · multibully_cpu · schema_multibully (+_memelabel) · smoke_*      scripts/  run_final.sh · get_glove.sh · download_data.sh
src/toxifusionnet/{preprocessing,data,models,training,explain,baselines,robustness,experiments,analysis,inference,app}/  cli.py
results/    RESULTS.md and all tables / figures (no post text)       models/  shipped small model        docs/  report (Word)
tests/ (48)         notebooks/ run_on_colab.ipynb
```

## Known limitations
Small non-pre-trained encoder; sarcasm labels depend partly on the image; significance tests ignore training randomness;
secondary ablations have one seed; heuristic language ID and seed lexicons; decision threshold fixed at 0.5.
Hate-speech classifiers err on dialect, reclaimed slurs and identity terms: use for human-review triage, not automatic sanctions.

## References
Maity, Jha, Saha, Bhattacharyya (2022), SIGIR '22 (MultiBully) · Mamun et al. (2024), Appl. Sci. 14(11):4898 · Mathew et al., HateXplain, AAAI 2021 ·
Pennington et al. (2014), GloVe · Khanuja et al., MuRIL, 2021 · Kendall et al., CVPR 2018 · DeYoung et al., ERASER, ACL 2020 · Ribeiro et al., LIME, KDD 2016.
