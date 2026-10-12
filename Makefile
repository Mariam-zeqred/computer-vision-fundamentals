.PHONY: check lint test lessons

PY ?= python3

check: lint test lessons

lint:
	ruff check .
	ruff format --check .

test:
	$(PY) -m pytest -q

# Every lesson script must run from a clean checkout and rewrite its figures.
lessons:
	$(PY) lesson1.py
