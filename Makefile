.PHONY: install lint format test train evaluate predict

install:
	pip install -r requirements.txt
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest

train:
	python -m storm_forecasting.cli.train --config configs/experiments/baseline_reproduction.yaml

evaluate:
	python -m storm_forecasting.cli.evaluate --config configs/experiments/baseline_reproduction.yaml

predict:
	python -m storm_forecasting.cli.predict --config configs/experiments/baseline_reproduction.yaml --checkpoint outputs/checkpoints/baseline_reproduction/best.pt --index 0
