import torch
import torch.nn as nn
from model import IrisClassifier
from train import test_loader


model = IrisClassifier()
model.load_state_dict(torch.load("models/iris_classifier.pt"))
model.eval()

def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

test_model(model, test_loader)
