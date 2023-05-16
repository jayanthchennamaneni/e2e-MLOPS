import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_iris_data
from model import IrisClassifier

train_loader, test_loader = load_iris_data()

model = IrisClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    """
    Training loop for the IrisClassifier model.

    Args:
    - epoch (int): Current epoch number.
    """
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/iris_classifier.pt")
