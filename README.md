# End-to-End MLOps Project

This repository contains an end-to-end Machine Learning Operations (MLOps) project for the classification of iris flowers using the famous Iris dataset. The project is built using PyTorch for model training, FastAPI for API development, Uvicorn as the server, and Docker for containerization.

## Project Overview

The main objective of this project is to implemnt a complete MLOps pipeline, including model training, testing, serving, and containerization. The project is organized into the following structure:

- `models/`: Directory for storing trained model files.
- `Dockerfile`: File to define the Docker container for running the training script.
- `Makefile`: File to define the build process and manage dependencies.
- `requirements.txt`: File listing the Python packages required for the project.
- `data_loader.py`: Python script for loading the dataset.
- `model.py`: Python script for defining the neural network. 
- `train.py`: Python script for training the model.
- `test.py`: Python script for testing the model.
- `app.py`: FastAPI endpoints definition


