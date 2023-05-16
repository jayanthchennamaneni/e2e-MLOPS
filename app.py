from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import IrisClassifier

app = FastAPI()

class IrisData(BaseModel):
    """
    Represents the input data for Iris flower prediction.

    Attributes:
    - sepal_length (float): Length of the sepal.
    - sepal_width (float): Width of the sepal.
    - petal_length (float): Length of the petal.
    - petal_width (float): Width of the petal.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

model = IrisClassifier()
model.load_state_dict(torch.load("models/iris_classifier.pt"))

@app.post("/predict")
async def predict(iris_data: IrisData):
    """
    Endpoint for predicting the class of an Iris flower based on the provided data.

    Args:
    - iris_data (IrisData): Input data containing sepal and petal measurements.

    Returns:
    - dict: A dictionary containing the predicted class of the Iris flower.
    """
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
