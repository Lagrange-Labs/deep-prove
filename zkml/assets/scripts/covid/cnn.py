import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import json

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
TRAIN_PATH = "data/Dataset/Train"
VAL_PATH = "data/Dataset/Val"

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define CNN model with stride=2
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Compute the output size dynamically
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self._forward_conv(dummy_input)
            self.flatten_size = dummy_output.view(1, -1).size(1)  # Compute flattened size

        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_conv(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.relu(self.conv5(x))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize model, loss, and optimizer
model = CNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Train the model
train(model, train_loader, criterion, optimizer, epochs=5)

# Save the trained model
torch.save(model.state_dict(), "cnn_model.pth")

# Export to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model, dummy_input, "cnn_model.onnx", input_names=["input"], output_names=["output"])

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

            test_data["input_data"].append(images.cpu().tolist())  # Raw input
            test_data["output_data"].append(labels.cpu().tolist())  # True labels
            test_data["pytorch_output"].append(outputs.cpu().tolist())  # Model predictions

    with open("test_data.json", "w") as f:
        json.dump(test_data, f, indent=4)

export_test_data(model, val_loader)
