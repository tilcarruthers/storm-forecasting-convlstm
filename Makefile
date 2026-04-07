PYTHON := .venv/bin/python
REPO_ID := benmoseley/ese-dl-2025-26-group-project
CONFIG := configs/experiments/baseline_reproduction.yaml

.PHONY: install lint format test download-data build-index train evaluate predict

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest

download-data:
	$(PYTHON) -m storm_forecasting.cli.make_dataset_index \
		--config $(CONFIG) \
		--download

build-index:
	$(PYTHON) -m storm_forecasting.cli.make_dataset_index \
		--config $(CONFIG)

train:
	$(PYTHON) -m storm_forecasting.cli.train --config $(CONFIG) --save-config-artifacts

evaluate:
	$(PYTHON) -m storm_forecasting.cli.evaluate --config $(CONFIG) --save-config-artifacts

predict:
	$(PYTHON) -m storm_forecasting.cli.predict --config $(CONFIG) --checkpoint outputs/checkpoints/baseline_reproduction/best.pt --index 0
