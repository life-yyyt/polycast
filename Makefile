.PHONY: install test lint eval serve clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

eval:
	polycast eval --dataset evals/questions.json

serve:
	polycast serve --reload

clean:
	rm -rf runs polycast.db .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
