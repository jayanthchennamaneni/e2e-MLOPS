from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import IrisClassifier
from test import accuracy

app = FastAPI()

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

model = IrisClassifier()
model.load_state_dict(torch.load("models/iris_classifier.pt"))

@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.

    Returns:
    - dict: A dictionary containing a welcome message.
    """
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict")
async def predict(iris_data: IrisData):
    data = torch.tensor([[
        iris_data.sepal_length,
        iris_data.sepal_width,
        iris_data.petal_length,
        iris_data.petal_width,
    ]])

    class_names = ['setosa', 'versicolor', 'virginica']

    model.eval()
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[int(predicted)]
        return {"class": predicted_class}
    
@app.get("/metrics")
async def metrics():
    """
    Endpoint that returns the model's performance metrics.

    Returns:
    - dict: A dictionary containing the model's performance metrics.
    """
    return {"accuracy": accuracy}
