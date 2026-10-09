.PHONY: build up test

IMAGE_NAME := trial-test-ai-ml-engineer

# Picked up automatically if you create a .env file (e.g. for OPENAI_API_KEY) --
# harmless if it doesn't exist.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

build:
	docker build -t $(IMAGE_NAME) .

up: build
	docker run --rm -it -p 5000:5000 -v "$(CURDIR)":/app -v $$HOME/.cache/huggingface:/root/.cache/huggingface $(ENV_FILE) $(IMAGE_NAME)

test: build
	docker run --rm -v "$(CURDIR)":/app -v $$HOME/.cache/huggingface:/root/.cache/huggingface $(ENV_FILE) $(IMAGE_NAME) pytest
