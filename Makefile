# PathSnap — Group 8
# Usage: make <target>

PYTHON ?= python3
DATA_ROOT ?= ./data
ARTIFACT_DIR ?= ./artifacts

.PHONY: help install test smoke eda dataset train tune eval-hmm export api ui status clean

help:
	@echo "PathSnap targets:"
	@echo "  make install     Install Python dependencies"
	@echo "  make test        Run unit tests (no dataset required)"
	@echo "  make smoke       Import the package and validate imports"
	@echo "  make status      Cross-agent status snapshot (git/data/model/viva)"
	@echo "  make eda         Dataset inventory + EDA figures  -> results/eda/"
	@echo "  make dataset     Build cached candidate/feature matrix (Phase 1)"
	@echo "  make train       Train and compare >=5 models     -> results/models/"
	@echo "  make tune        Bayesian hyperparameter search   -> results/optuna/"
	@echo "  make eval-hmm    Classifier vs HMM comparison     -> results/hmm/"
	@echo "  make export      Export best model + configs      -> artifacts/"
	@echo "  make api         Run FastAPI on 0.0.0.0:8000 (Laptop 1)"
	@echo "  make ui          Run Streamlit on 0.0.0.0:8501 (Laptop 2)"
	@echo "  make clean       Remove caches and generated temp files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -c "import pathsnap, pathsnap.io, pathsnap.geo, pathsnap.network, pathsnap.candidates, pathsnap.metrics, pathsnap.hmm; print('imports OK')"

status:
	$(PYTHON) scripts/agent_status.py

eda:
	$(PYTHON) scripts/01_eda.py --data-root $(DATA_ROOT)

dataset:
	$(PYTHON) scripts/02_build_dataset.py --data-root $(DATA_ROOT)

train:
	$(PYTHON) scripts/03_train_models.py --data-root $(DATA_ROOT)

tune:
	$(PYTHON) scripts/04_tune.py --data-root $(DATA_ROOT)

eval-hmm:
	$(PYTHON) scripts/05_eval_hmm.py --data-root $(DATA_ROOT)

export:
	$(PYTHON) scripts/06_export_artifacts.py --artifact-dir $(ARTIFACT_DIR)

api:
	PATHSNAP_DATA_ROOT=$(DATA_ROOT) $(PYTHON) -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info

ui:
	$(PYTHON) -m streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501

clean:
	rm -rf .pytest_cache **/__pycache__ cache/ .cache/
	find . -name "*.pyc" -delete
