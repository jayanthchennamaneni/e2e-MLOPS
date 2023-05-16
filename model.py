import torch
import torch.nn as nn

class IrisClassifier(nn.Module):
    """
    A neural network model for classifying Iris flowers.

    Architecture:
    - Input layer: 4 units
    - Hidden layer 1: 16 units, ReLU activation
    - Hidden layer 2: 8 units, ReLU activation
    - Output layer: 3 units
    """

    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        """
        Performs forward pass of the IrisClassifier model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, 4)

        Returns:
        - output (torch.Tensor): Output tensor of shape (batch_size, 3)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
