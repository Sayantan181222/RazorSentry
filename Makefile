.PHONY: install eval test run docker-build docker-up

install:
	pip install -r requirements.txt

eval:
	python src/eval.py

test:
	pytest tests/ -v

run:
	uvicorn src.service:app --reload --port 8000

docker-build:
	docker build -t razorsentry .

docker-up:
	docker compose up --build
