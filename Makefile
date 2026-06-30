.PHONY: install install-db test lint run example docker-up docker-down

install:
	pip install -r requirements.txt

install-db:
	pip install -r requirements.txt -r requirements-db.txt

test:
	pytest

lint:
	ruff check app

run:
	uvicorn app.main:app --reload

example:
	python examples/quickstart.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
