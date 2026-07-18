# Common developer tasks. Run `make help` for the list.

.PHONY: help install demo acl-demo structured-demo fatslim-demo flat-demo test lint format format-check typecheck redis-up redis-down check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras (pinned versions via uv.lock)
	uv sync --extra dev

demo:  ## Run the scripted end-to-end demo (fake providers, in-memory store)
	uv run python -m meridian.interfaces.cli.main --demo

acl-demo:  ## Show the access-control retrieval filter in isolation
	uv run python -m meridian.interfaces.cli.main --acl-demo

structured-demo:  ## Show the structured query path compiling to RediSearch
	uv run python -m meridian.interfaces.cli.main --structured-demo

fatslim-demo:  ## Show the fat/slim retrieval split (slim search, fat fetch)
	uv run python -m meridian.interfaces.cli.main --fatslim-demo

flat-demo:  ## Show the flat, pre fat/slim knowledge-chunk model, for contrast
	uv run python -m meridian.interfaces.cli.main --flat-demo

test:  ## Run the test suite
	uv run pytest

lint:  ## Lint with ruff
	uv run ruff check src tests

format:  ## Format with ruff
	uv run ruff format src tests

format-check:  ## Verify formatting without changing files
	uv run ruff format --check src tests

typecheck:  ## Type-check with mypy
	uv run mypy

check: format-check lint typecheck test  ## Run format, lint, typecheck, and tests

redis-up:  ## Start Redis Stack via Docker Compose
	docker compose up -d

redis-down:  ## Stop Redis Stack
	docker compose down
