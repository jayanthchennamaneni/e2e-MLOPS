# End-to-End MLOps Project

This repository contains an end-to-end Machine Learning Operations (MLOps) project for the classification of iris flowers using the famous Iris dataset. The project is built using PyTorch for model training, FastAPI for API development, Uvicorn as the server, and Docker for containerization.

## Project Overview

The main objective of this project is to implment a complete MLOps pipeline, including model training, testing, containerization, and serving. The project is organized into the following structure:

- `models/`: Directory for saved model files
- `Dockerfile`: Dockerfile for containerization
- `Makefile`: File to define the build process and manage dependencies
- `requirements.txt`: Project dependencies
- `data_loader.py`: Python script for loading the dataset
- `model.py`: simple feed-forward neural network model definition 
- `train.py`: Model training script
- `test.py`: Model testing script
- `app.py`: FastAPI endpoints definition


