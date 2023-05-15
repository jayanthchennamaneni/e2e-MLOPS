# End-to-End MLOps Project

This repository contains an example of an end-to-end Machine Learning Operations (MLOps) project using Docker, Makefile, and PyTorch. The goal of this project is to training, and deploying a simple machine learning model for a classification task.

## Project Overview

This project uses the Iris dataset to train a neural network for classifying Iris flower species. The project is organized into the following structure:

- `models/`: Directory for storing trained model files.
- `Dockerfile`: File to define the Docker container for running the training script.
- `Makefile`: File to define the build process and manage dependencies.
- `requirements.txt`: File listing the Python packages required for the project.
- `data_loader.py`: Python script for loading the dataset.
- `model.py`: Python script for defining the neural network. 
- `train.py`: Python script for training the model.
- `test.py`: Python script for testing the model.
