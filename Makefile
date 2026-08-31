.PHONY: install eval test run run-prod docker-build docker-up docker-down docker-logs db-shell

install:
	pip install -r requirements.txt

eval:
	python src/eval.py

test:
	pytest tests/ -v

run:
	uvicorn src.service:app --reload --port 8000

run-prod:
	uvicorn src.service:app --host 0.0.0.0 --port 8000 --workers 4

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f razorsentry

db-shell:
	docker compose exec db psql -U razorsentry -d razorsentry
