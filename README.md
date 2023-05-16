# End-to-End MLOps Project

This repository contains an end-to-end Machine Learning Operations (MLOps) project for the classification of iris flowers using the famous Iris dataset. The project is built using PyTorch for model training, FastAPI for API development, Uvicorn as the server, and Docker for containerization.

## Project Overview

The main objective of this project is to develop a machine learning model for the classification of iris flowers based on their features, such as sepal length, sepal width, petal length, and petal width. The project also focuses on implementing a complete MLOps pipeline, including model training, testing, serving, and containerization. The project is organized into the following structure:

.
├── models                # Directory for saved model files
├── .dockerignore         # Docker ignore file
├── .gitignore            # Git ignore file
├── dataloader.py         # Data loading and dataloaders using PyTorch
├── model.py              # Neural network model definition
├── train.py              # Model training script
├── test.py               # Model testing script
├── app.py                # FastAPI endpoints definition
├── requirements.txt      # Project dependencies
├── Makefile              # defining the build process
└── Dockerfile            # Dockerfile for containerization

