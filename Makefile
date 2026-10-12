PY ?= python
export PYTHONPATH := src
CFG ?= configs/multibully.yaml
RUN ?= runs/multibully/full/seed42
ENC ?= google/muril-base-cased

.PHONY: final-cpu describe final smoke-multibully install test smoke inspect prepare train baselines ablate robust explain bench figures serve clean

install:            ## editable install with all extras
	pip install -e ".[explain,serve,dev]"
test:               ## unit + integration tests (offline)
	$(PY) -m pytest -q
smoke:              ## offline end-to-end smoke test on SYNTHETIC data (tiny random BERT) -- mechanics only
	$(PY) -m toxifusionnet.cli synth --n 3000 --out data/processed/synthetic.csv
	$(PY) -m toxifusionnet.cli train --config configs/smoke_test.yaml --run-dir runs/smoke
	$(PY) -m toxifusionnet.cli predict --run-dir runs/smoke --text "Wah wah, kya baat hai, great job lol"
final-cpu:           ## RAW=data/raw/multibully.csv  CPU study with the small GloVe-initialised encoder (run scripts/get_glove.sh first)
	CFG=configs/multibully_cpu.yaml CV=1 bash scripts/run_final.sh $(RAW)
smoke-multibully:   ## offline smoke test of the REAL-DATA path: MultiBully-layout xlsx (synthetic rows) -> prepare -> train
	$(PY) -m toxifusionnet.cli synth --layout multibully --n 1500 --out data/raw/multibully_synth.csv
	$(PY) -m toxifusionnet.cli prepare --schema configs/schema_multibully.yaml --raw data/raw/multibully_synth.csv --out data/processed/multibully_synth.csv
	$(PY) -m toxifusionnet.cli describe --data data/processed/multibully_synth.csv --out results/smoke_data_report.md
	$(PY) -m toxifusionnet.cli train --config configs/smoke_multibully.yaml --run-dir runs/smoke_multibully
final:              ## RAW=data/raw/<MultiBully file>  [ENC=google/muril-base-cased]  full study -> results/RESULTS.md
	bash scripts/run_final.sh $(RAW) $(ENC)
describe:           ## DATA=data/processed/multibully.csv
	$(PY) -m toxifusionnet.cli describe --data $(DATA)
inspect:            ## RAW=data/raw/file.csv  -> columns, label counts, schema draft
	$(PY) -m toxifusionnet.cli inspect --raw $(RAW)
prepare:            ## RAW=... raw + configs/schema_multibully.yaml -> data/processed/multibully.csv
	$(PY) -m toxifusionnet.cli prepare --schema configs/schema_multibully.yaml --raw $(RAW) --out data/processed/multibully.csv
train:
	$(PY) -m toxifusionnet.cli train --config $(CFG) --run-dir $(RUN)
baselines:
	$(PY) -m toxifusionnet.cli baselines --data data/processed/dataset.csv --out runs/baselines/baselines.csv
ablate:             ## all ablations + standard-transformer baseline, 3 seeds
	$(PY) -m toxifusionnet.cli ablate --config $(CFG) --out-dir runs/ablation --seeds 42 43 44
robust:
	$(PY) -m toxifusionnet.cli robust --run-dir $(RUN)
explain:
	$(PY) -m toxifusionnet.cli explain --run-dir $(RUN)
bench:
	$(PY) -m toxifusionnet.cli bench --run-dir $(RUN)
figures:
	$(PY) -m toxifusionnet.cli figures --run-dir $(RUN)
serve:
	$(PY) -m toxifusionnet.cli serve --run-dir $(RUN) --port 8000
clean:
	rm -rf runs .pytest_cache
