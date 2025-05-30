#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
import argparse

# Add argument parser similar to mlp.py
parser = argparse.ArgumentParser(description="CIFAR10 CNN with ONNX export")
parser.add_argument("--export",
                    type=Path,
                    default=Path('bench'),
                    help="Directory to export the model to (default: bench)")
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Function to adapt paths dynamically
def get_data_path(subdir):
    return os.path.join(script_dir, subdir)


# Directories (now using dynamic paths)
TRAIN_PATH = get_data_path("data/Dataset/Train")
VAL_PATH = get_data_path("data/Dataset/Val")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
])

# Load dataset
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.conv5 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=0)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        # Flatten (handled in forward pass)
        # Dense layers
        self.fc1 = nn.Linear(18432, 64)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(64, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional and pooling layers with dropout
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout1(x)
        # Flatten
        # x = x.view(x.size(0), -1)  # Flatten dynamically based on batch size
        x = torch.flatten(x, 1)

        # Dense layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        # x = self.sigmoid(self.fc2(x))
        x = self.relu(self.fc2(x))

        return x


# Initialize model, loss, and optimizer
model = CNNModel().to(device)
# model = CNN2().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(
                device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}"
        )


# Train the model
train(model, train_loader, criterion, optimizer, epochs=5)

# Save the trained model
torch.save(model.state_dict(), "cnn_model.pth")

# Export to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)
model_path = args.export / "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    model_path,
    export_params=True,
    # do_constant_folding=True,
    opset_version=12,
    input_names=["input"],
    output_names=["output"])

# dynamic_axes={'input': {0: 'batch_size'},
#       'output': {0: 'batch_size'}})


# Generate test data JSON
def export_test_data(model, val_loader, num_samples=5):
    model.eval()
    test_data = {"input_data": [], "output_data": [], "pytorch_output": []}
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= num_samples:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            test_data["input_data"].append(images.cpu().reshape(
                [-1]).tolist())  # Raw input
            test_data["output_data"].append(labels.cpu().reshape(
                [-1]).tolist())  # True labels
            test_data["pytorch_output"].append(outputs.cpu().reshape(
                [-1]).tolist())

    data_path = args.export / "input.json"
    with open(data_path, "w") as f:
        json.dump(test_data, f, indent=4)


export_test_data(model, val_loader)
