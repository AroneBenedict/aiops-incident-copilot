.PHONY: install test lint evaluate serve

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest --cov --cov-report=term-missing

evaluate:
	python -m aiops.pipeline --seeds 7 11 13 17 19 --out results

serve:
	uvicorn aiops.api:app --port 8000
