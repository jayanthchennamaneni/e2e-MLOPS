.PHONY: train build-docker

train:
	python train.py

build-docker:
	docker build -t toast .
