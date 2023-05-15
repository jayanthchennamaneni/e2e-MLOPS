.PHONY: train test build-docker

train:
	python train.py

test:
	python test.py

build-docker:
	docker build -t toast .
