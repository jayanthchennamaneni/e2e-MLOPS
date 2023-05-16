.PHONY: train test build-docker

train:
	# Run the training script
	python train.py

test:
	# Run the testing script
	python test.py

build-docker:
	# Build a Docker image with the tag "toast"
	docker build -t toast .

