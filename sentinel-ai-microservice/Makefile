.PHONY: install dev test lint run train docker-build docker-run compose-up compose-down k8s-deploy k8s-delete clean

IMAGE ?= sentinel-ai-microservice:local

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest --cov=app --cov-report=term-missing

lint:
	ruff check .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

train:
	python scripts/train_model.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

compose-up:
	docker compose up --build

compose-monitoring:
	docker compose --profile monitoring up --build

compose-down:
	docker compose down

k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl -n sentinel apply -f k8s/service.yaml
	kubectl -n sentinel apply -f k8s/hpa.yaml
	kubectl -n sentinel apply -f k8s/deployment.yaml
	kubectl -n sentinel apply -f k8s/ingress.yaml

k8s-delete:
	kubectl delete namespace sentinel --ignore-not-found

clean:
	rm -rf app/artifacts .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
