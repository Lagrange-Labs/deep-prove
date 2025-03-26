#!/usr/bin/env python

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import argparse
from pathlib import Path
import torch.optim as optim

parser = argparse.ArgumentParser(description="mlp generator --num-dense and --layer-width")
parser.add_argument("--num-dense", type=int, required=True, help="Number of dense layers")
parser.add_argument("--layer-width", type=int, required=True, help="Width of each layer")
parser.add_argument("--export", type=Path, default=Path('bench'), help="Directory to export the model to (default: bench)")
parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples to export")
parser.add_argument("--distribution", action="store_true", help="Show distribution of model weights")

args = parser.parse_args()
print(f"num_dense: {args.num_dense}, layer_width: {args.layer_width}")
# Ensure the folder exists
if not args.export.exists() or not args.export.is_dir():
    print(f"❌ Error: export folder '{args.export}' does not exist or is not a directory.")
    exit(1)


# Load the iris data
iris = load_iris()
dataset = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target'])
print("Loaded iris data")


class MLP(nn.Module):
    def __init__(self, num_dense, layer_width):
        super(MLP, self).__init__()
        layers = []
        input_size = 4  # Assuming input size is 4 for the Iris dataset
        for _ in range(num_dense):
            layers.append(nn.Linear(input_size, layer_width, bias=True))
            layers.append(nn.ReLU())
            input_size = layer_width
        layers.append(nn.Linear(layer_width, 3, bias=True))  # Assuming 3 output classes
        layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Create model
model = MLP(num_dense=args.num_dense, layer_width=args.layer_width)

# Extract input features
X = dataset[dataset.columns[0:4]].values
y = dataset.target

# Normalize inputs to [-1,1] range
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

# Train-test split after normalization
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2
)
print("Divided the data into testing and training.")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

EPOCHS = 800


print("Convert to pytorch tensor.")
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y.values).long())
test_y = Variable(torch.Tensor(test_y.values).long())


loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))


for epoch in tqdm.trange(EPOCHS):

    predicted_y = model(train_X)
    loss = loss_fn(predicted_y, train_y)
    loss_list[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_pred = model(test_X)
        correct = (torch.argmax(y_pred, dim=1) ==
                   test_y).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list)
ax1.set_ylabel("Accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("Loss")
ax2.set_xlabel("epochs")
plt.tight_layout(pad=0.08)
plt.savefig("accuracy-loss.png")


x = test_X[0].reshape(1, 4)
model.eval()

y_pred = model(test_X[0])
print("Expected:", test_y[0], "Predicted", torch.argmax(y_pred, dim=0))

from pathlib import Path

model_path = args.export / "model.onnx"
data_path = args.export / "input.json"

x = test_X[0].reshape(1, 4)
model.eval()
torch.onnx.export(model,
                  x,
                  model_path,
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

print(f"Model onnx exported to {model_path}")

# Use the first `num_samples` test samples
num_samples = min(args.num_samples, len(test_X))

# Prepare data arrays
input_data = []
output_data = []
pytorch_output = []

# Process each selected test sample
for i in range(num_samples):
    x = test_X[i]
    true_label = test_y[i].item()
    
    # Evaluate model
    with torch.no_grad():
        model_output = model(x.unsqueeze(0)).squeeze(0)
    
    # Get input data
    input_data.append(x.detach().numpy().reshape([-1]).tolist())
    
    # Create one-hot encoded ground truth output
    one_hot_output = [0.0, 0.0, 0.0]  # For 3 classes in Iris
    one_hot_output[true_label] = 1.0
    output_data.append(one_hot_output)
    pytorch_output.append(model_output.tolist())

# Save multiple input/output pairs to JSON
data = {
    "input_data": input_data, 
    "output_data": output_data, 
    "pytorch_output": pytorch_output
}
json.dump(data, open(data_path, 'w'), indent=2)
print(f"Input/Output data for {num_samples} samples exported to {data_path}")

def tensor_to_vecvec(tensor):
    """Convert a PyTorch tensor to a Vec<Vec<_>> format and print it."""
    vecvec = tensor.tolist()
    for i, row in enumerate(vecvec):
        formatted_row = ", ".join(f"{float(val):.2f}" for val in row)
        print(f"{i}: [{formatted_row}]")

# Print the weight matrices in Vec<Vec<_>> format and their dimensions
for i, layer in enumerate(model.layers):
    if isinstance(layer, nn.Linear):
        weight_matrix = layer.weight.data
        bias_vector = layer.bias.data
        print(f"Layer {i} weight matrix dimensions: {weight_matrix.size()}")
        print(f"Layer {i} bias vector dimensions: {bias_vector.size()}")

def plot_weight_distribution(model):
    """Plot the distribution of weights across all dense layers."""
    # Collect all weights from dense layers
    all_weights = []
    layer_names = []
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            # Flatten weights and add to list
            weights = layer.weight.data.cpu().numpy().flatten()
            all_weights.append(weights)
            layer_names.append(f"Layer {i}")
    
    # Create subplots for each layer
    fig, axes = plt.subplots(len(all_weights), 1, figsize=(12, 4*len(all_weights)))
    if len(all_weights) == 1:
        axes = [axes]
    
    for ax, weights, name in zip(axes, all_weights, layer_names):
        # Calculate min and max
        w_min = np.min(weights)
        w_max = np.max(weights)
        
        # Add 5% padding to the range
        range_pad = (w_max - w_min) * 0.05
        x_min = w_min - range_pad
        x_max = w_max + range_pad
        
        # Plot histogram with dynamic range
        ax.hist(weights, bins=50, density=False, alpha=0.7, label='Count', range=(x_min, x_max))
        
        ax.set_title(f'Weight Distribution - {name}\nMin: {w_min:.3f}, Max: {w_max:.3f}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_xlim(x_min, x_max)  # Set dynamic x-axis limits
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Position legend outside the plot
        ax.grid(True)
    
    plt.tight_layout()
    # Save the figure with extra space for legend
    plt.savefig("weight_distribution.png", bbox_inches='tight', dpi=300)
    # Show the plot interactively
    plt.show()
    # Close the figure to free memory
    plt.close()

# After training loop, before ONNX export
if args.distribution:
    print("Generating weight distribution plot...")
    plot_weight_distribution(model)
    print("Weight distribution plot saved as 'weight_distribution.png'")