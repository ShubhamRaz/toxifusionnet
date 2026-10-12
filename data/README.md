# Data

| Folder | Content |
|---|---|
| `raw/` | Original downloads, never edited |
| `processed/` | Canonical CSVs written by `toxifusion prepare` |

## MultiBully (the real dataset used by this project)
*Source.* Maity, Jha, Saha, Bhattacharyya, **A Multitask Framework for Sentiment, Emotion and Sarcasm aware Cyberbullying
Detection from Multi-modal Code-Mixed Memes**, SIGIR 2022. Official repo <https://github.com/Jhaprince/MultiBully> (code +
links to the annotation sheet and the meme images). The Kaggle copy <https://www.kaggle.com/datasets/pranee5/multi-bully>
is assumed to be the same data -- confirm with `make inspect`.

*What you need.* Only the **annotation sheet** (`.xlsx`/`.csv`, a few hundred KB). The meme **images are not used** (this is a
text-only model). Columns of the official sheet and how they are used:

| Column | Values | Use in ToxiFusionNet |
|---|---|---|
| `Img-Text` | OCR text of the meme | **model input** |
| `Text-Label` | Bully / Nonbully (text alone) | **hate** label (default; see below) |
| `Img-Text-Label` | Bully / Nonbully (image + text) | alternative hate label (meme level) |
| `Img-Label` | Bully / Nonbully (image alone) | not used |
| `Sarcasm` | Yes / No | **sarcasm** label (gold) |
| `Harmful-Score` | Harmless / Partially-Harmful / Very-Harmful | **severity**: low / moderate / high |
| `Target` | Individual / Organization / Community / Society / None | **target** (4-class taxonomy) |
| `Sentiment`, `Emotion` | see paper | not used (possible auxiliary tasks) |
| `Doubtful` | annotator flag | optional exclusion (`exclude:` in the schema) |

*Decisions you should know about* (all configurable in `configs/schema_multibully.yaml`):
1. **Text-Label vs. Img-Text-Label.** A text-only model can only see the text, so the default hate label is `Text-Label`.
   `Img-Text-Label` is the dataset's headline (meme-level) label; many of its "Bully" labels are caused by the image.
2. **Target taxonomy.** The flow chart lists individual/caste/religion/gender/demography; MultiBully annotates
   individual/organization/community/society. The two cannot be mapped without inventing information, so the 4-class
   taxonomy is used (`labels.target`). To keep the flow-chart taxonomy you must annotate the target yourself.
3. **Severity** is the harmfulness rating of the whole meme (image + text).
4. **Language.** The memes are not all Hinglish -- inspect with `make inspect` / `toxifusion describe`.

## Getting the file
The archive supplied for this project contains ~6,000 images and `Cyberbully_corrected_emotion_sentiment.xlsx - cyberbully.csv` (5,865 rows, 5,854 labelled). Copy that csv to `data/raw/multibully.csv`.

```bash
bash scripts/download_data.sh      # Kaggle API token needed; or download the sheet manually from the official repo README
make inspect RAW=data/raw/<file>.xlsx
make final   RAW=data/raw/<file>.xlsx          # whole study -> results/RESULTS.md
```

## Canonical format (`data/processed/*.csv`)
`id, text, sarcasm, hate, target, severity, sarcasm_rationale, hate_rationale, narrative, label_source, split` -- labels are
readable strings; an empty label is **masked** in the loss; target/severity are empty for non-hate posts; `label_source`
records `gold` / `gold-derived` / `weak` / `synthetic` per task.
