# MLOps with Iris Classification

This repository demonstrates an end-to-end Machine Learning Operations (MLOps) pipeline for Iris flower classification using PyTorch, FastAPI, Uvicorn, and Docker. It serves as a prototype for managing the lifecycle of a machine learning model, including training, testing, containerization, and deployment.

# Project Overview

The primary objective is to showcase an effective MLOps pipeline. The process starts with a feed-forward neural network model that is trained and tested using the renowned Iris dataset. The trained model is then served using a FastAPI application running on a Uvicorn server, and the application is finally containerized using Docker.


# Repository Structure 

The repository's structure is as follows:

````
├── models              # Directory for trained models
├── Dockerfile          # Dockerfile for containerizing the application
├── Makefile            # Automates the build process and manages dependencies
├── requirements.txt    # Required Python dependencies
├── data_loader.py      # Script to load the Iris dataset
├── model.py            # Feed-forward neural network model definition
├── train.py            # Script to train the model
├── test.py             # Script to test the model
└── app.py              # FastAPI application with endpoints to interact with the model
```


## Getting Started

To get the project up and running:

1. Clone the repository.
2. Install the necessary Python packages with `pip install -r requirements.txt`.
3. Run the `Makefile` to train and test the model, and build the Docker image: `make all`.

## Running the FastAPI microservices

Once you have built the Docker image, you can start the FastAPI application:

```
docker run -p 8000:8000 <docker-image-name>:<tag>
```


Then, access the API at http://localhost:8000.

## Additional Resources

- [PyTorch Documentation](https://pytorch.org)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [uvicorn Documentation](https://www.uvicorn.org)
- [Docker Documentation](https://docs.docker.com/)

## License

This project is licensed under the MIT License. See the [LICENSE] file for details.
