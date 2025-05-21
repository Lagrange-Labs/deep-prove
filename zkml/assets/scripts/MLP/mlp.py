#!/usr/bin/env python

from torch.ao.quantization import MinMaxObserver
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
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.utils.data import Subset
import math  # Add to imports at top

print(torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = 'qnnpack'

parser = argparse.ArgumentParser(
    description="mlp generator --num-dense and --layer-width")
parser.add_argument("--num-dense", type=int, required=True,
                    help="Number of dense layers")
parser.add_argument("--layer-width", type=int,
                    required=True, help="Width of each layer")
parser.add_argument("--export", type=Path, default=Path('bench'),
                    help="Directory to export the model to (default: bench)")
parser.add_argument("--num-samples", type=int, default=100,
                    help="Number of test samples to export")
parser.add_argument("--distribution", action="store_true",
                    help="Show distribution of model weights")
parser.add_argument("--no-relu", action="store_true",
                    help="Disable ReLU activation functions")

args = parser.parse_args()
print(f"num_dense: {args.num_dense}, layer_width: {args.layer_width}")
# Ensure the folder exists
if not args.export.exists() or not args.export.is_dir():
    print(
        f"❌ Error: export folder '{args.export}' does not exist or is not a directory.")
    exit(1)


# Load the iris data
iris = load_iris()
dataset = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target'])
print("Loaded iris data")


class MLP(nn.Module):
    def __init__(self, num_dense, layer_width, quantize=False, use_relu=True):
        super(MLP, self).__init__()
        if quantize:
            self.quant = QuantStub()
        self.quantize = quantize
        self.use_relu = use_relu
        self.useBias = True
        if quantize:
            self.dequant = DeQuantStub()

        layers = []
        input_size = 4  # Assuming input size is 4 for the Iris dataset
        for _ in range(num_dense):
            layers.append(
                nn.Linear(input_size, layer_width, bias=self.useBias))
            if use_relu:
                layers.append(nn.ReLU())
            input_size = layer_width
        # Assuming 3 output classes
        layers.append(nn.Linear(layer_width, 3, bias=self.useBias))
        if use_relu:
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        for layer in self.layers:
            x = layer(x)
        if self.quantize:
            x = self.dequant(x)
        return x


# Modify the model creation line to enable quantization
model = MLP(num_dense=args.num_dense, layer_width=args.layer_width,
            quantize=True, use_relu=True)

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
        print(f"Layer {i} weight matrix dimensions: {weight_matrix.size()}")
        if model.useBias:
            bias_vector = layer.bias.data
            print(f"Layer {i} bias vector dimensions: {bias_vector.size()}")


def plot_weight_distribution(model, test_X):
    """Plot the distribution of weights, biases, and test inputs."""
    # Collect all weights and biases from dense layers
    all_weights = []
    all_biases = []
    layer_names = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            # Flatten weights and biases and add to lists
            weights = layer.weight.data.cpu().numpy().flatten()
            biases = layer.bias.data.cpu().numpy().flatten()
            all_weights.append(weights)
            all_biases.append(biases)
            layer_names.append(f"Layer {i}")

    # Create subplots: one row for inputs, then one row per layer
    num_layers = len(all_weights)
    fig, axes = plt.subplots(
        num_layers + 1, 2, figsize=(10, 3*(num_layers + 1)))
    if num_layers == 0:
        axes = axes.reshape(1, -1)

    # Plot test input distributions
    input_names = ['sepal length', 'sepal width',
                   'petal length', 'petal width']
    for i in range(4):
        input_data = test_X[:, i].numpy()
        i_min, i_max = np.min(input_data), np.max(input_data)
        i_range_pad = (i_max - i_min) * 0.05
        i_x_min, i_x_max = i_min - i_range_pad, i_max + i_range_pad

        axes[0, 0].hist(input_data, bins=50, density=False, alpha=0.7,
                        label=input_names[i], range=(i_x_min, i_x_max))

    axes[0, 0].set_title('Test Input Distribution\nAll Features')
    axes[0, 0].set_xlabel('Input Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot individual feature distributions
    for i in range(4):
        input_data = test_X[:, i].numpy()
        i_min, i_max = np.min(input_data), np.max(input_data)
        i_range_pad = (i_max - i_min) * 0.05
        i_x_min, i_x_max = i_min - i_range_pad, i_max + i_range_pad

        axes[0, 1].hist(input_data, bins=50, density=False, alpha=0.7,
                        label=input_names[i], range=(i_x_min, i_x_max))

    axes[0, 1].set_title('Test Input Distribution\nIndividual Features')
    axes[0, 1].set_xlabel('Input Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot weights and biases for each layer
    for ax_row, weights, biases, name in zip(axes[1:], all_weights, all_biases, layer_names):
        # Plot weights
        w_min, w_max = np.min(weights), np.max(weights)
        w_range_pad = (w_max - w_min) * 0.05
        w_x_min, w_x_max = w_min - w_range_pad, w_max + w_range_pad

        ax_row[0].hist(weights, bins=50, density=False, alpha=0.7,
                       label='Count', range=(w_x_min, w_x_max))
        ax_row[0].set_title(
            f'Weights - {name}\nMin: {w_min:.3f}, Max: {w_max:.3f}')
        ax_row[0].set_xlabel('Weight Value')
        ax_row[0].set_ylabel('Count')
        ax_row[0].set_xlim(w_x_min, w_x_max)
        ax_row[0].grid(True)

        # Plot biases
        b_min, b_max = np.min(biases), np.max(biases)
        b_range_pad = (b_max - b_min) * 0.05
        b_x_min, b_x_max = b_min - b_range_pad, b_max + b_range_pad

        ax_row[1].hist(biases, bins=50, density=False, alpha=0.7,
                       label='Count', range=(b_x_min, b_x_max))
        ax_row[1].set_title(
            f'Biases - {name}\nMin: {b_min:.3f}, Max: {b_max:.3f}')
        ax_row[1].set_xlabel('Bias Value')
        ax_row[1].set_ylabel('Count')
        ax_row[1].set_xlim(b_x_min, b_x_max)
        ax_row[1].grid(True)

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
    plot_weight_distribution(model, test_X)
    print("Weight distribution plot saved as 'weight_distribution.png'")

# Replace the DataLoader and evaluation sections with:


def evaluate_accuracy(model, test_X, test_y, num_samples):
    model.eval()
    correct = 0
    total = 0
    # Limit num_samples to the actual size of test data
    num_samples = min(num_samples, len(test_X))
    with torch.no_grad():
        for i in range(num_samples):
            x = test_X[i]
            true_label = test_y[i].item()
            output = model(x.unsqueeze(0)).squeeze(0)
            predicted = torch.argmax(output).item()
            if predicted == true_label:
                correct += 1
            total += 1
    return 100 * correct / total


# Replace the final evaluation with:
actual_samples = min(args.num_samples, len(test_X))
print(f"\nEvaluation accuracy on {actual_samples} samples: ", end='')
accuracy = evaluate_accuracy(model, test_X, test_y, actual_samples)
print(f"{accuracy:.2f}%")

# Add after the accuracy evaluation

# Verify accuracy consistency with JSON output
print("\nVerifying accuracy consistency...")
try:
    with open(data_path, 'r') as f:
        json_data = json.load(f)

    json_correct = 0
    json_total = 0

    for truth, pred in zip(json_data['output_data'], json_data['pytorch_output']):
        # Get predicted class from model output
        pred_class = max(range(len(pred)), key=lambda i: pred[i])
        # Get true class from one-hot encoded truth
        true_class = truth.index(1.0)

        if pred_class == true_class:
            json_correct += 1
        json_total += 1
        print(
            f"Run {json_total}: {pred_class == true_class}\n\t truth {truth}\n\t pytorch {pred}")

    json_accuracy = 100 * json_correct / json_total

    print(f"Accuracy from JSON file: {json_accuracy:.2f}%")

    # Compare rounded accuracies
    if round(json_accuracy) != round(accuracy):
        print(f"⚠️  WARNING: Accuracy mismatch!")
        print(f"    Direct evaluation: {round(accuracy)}%")
        print(f"    JSON file evaluation: {round(json_accuracy)}%")
        print(f"    This suggests an inconsistency in the evaluation process.")
except Exception as e:
    print(f"Error verifying accuracy consistency: {e}")

# calibration and quantization


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def evaluate(model, input, target, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    cnt = 0
    num_samples = len(test_X)
    with torch.no_grad():
        for i in range(num_samples):
            x = test_X[i]
            true_label = test_y[i].item()
            output = model(x.unsqueeze(0)).squeeze(0)
            predicted = torch.argmax(output).item()
            correct = 0
            if predicted == true_label:
                correct += 100
            cnt += 1
            print('.', end='')
            top1.update(correct)
            if cnt >= neval_batches:
                return top1

    return top1


def print_model_io(model, sample_input):
    inputs_by_layer = {}
    outputs_by_layer = {}

    def hook_fn(layer, inp, out):
        inputs_by_layer[layer] = inp[0].detach()  # Input
        outputs_by_layer[layer] = out.detach()    # Output

    hooks = []

    std_layers = (nn.Conv2d, nn.Linear)
    quantized_layers = (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)
    for layer in model.modules():
        if isinstance(layer, std_layers) or isinstance(layer, quantized_layers):
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        _ = model(sample_input)

    for i, layer in enumerate(model.modules()):
        if isinstance(layer, std_layers) or isinstance(layer, quantized_layers):
            layer_name = f"Layer {i} - {layer.__class__.__name__}"
            print(f"→ {layer_name}")

            # Input
            if isinstance(layer, quantized_layers):
                print("  Input:", inputs_by_layer.get(layer).int_repr())
                print("  Input scale:", inputs_by_layer.get(layer).q_scale())
            else:
                print("  Input:", inputs_by_layer.get(layer))

            # Weights
            if hasattr(layer, "weight"):
                try:
                    weight = layer.weight() if callable(layer.weight) else layer.weight.data
                except Exception:
                    weight = layer._weight_bias()[0]
                print("  Weights:")

                if isinstance(layer, quantized_layers):
                    print(weight.int_repr())
                    if hasattr(weight, 'q_scale'):
                        print("  Weight scale:", weight.q_scale())
                    if hasattr(weight, 'q_zero_point'):
                        print("  Weight zero_point:", weight.q_zero_point())
                else:
                    print(weight)
            # Bias
            if hasattr(layer, "bias") and layer.bias is not None:
                try:
                    bias = layer.bias() if callable(layer.bias) else layer.bias.data
                except Exception:
                    bias = layer._weight_bias()[1]
                print("  Bias:")
                print(bias)

            # Output
            if isinstance(layer, quantized_layers):
                print("  Output:", outputs_by_layer.get(layer).int_repr())
                print("  Output scale:", outputs_by_layer.get(layer).q_scale())
            else:
                print("  Output:", outputs_by_layer.get(layer))

            print("—" * 60)

        for h in hooks:
            h.remove()


sample_input = test_X[0].clone().detach().float().unsqueeze(0)

print_model_io(model, sample_input)

model.eval()


# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myconfig = torch.ao.quantization.qconfig.QConfig(
    activation=MinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_symmetric),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
model.qconfig = myconfig
print(model.qconfig)
torch.ao.quantization.prepare(model, inplace=True)
print("model prepared", model)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')

# Calibrate with the training set
# Create a subset of test data with exactly the same samples used in the JSON
json_sample_loader = torch.utils.data.DataLoader(
    test_X, batch_size=1, shuffle=False)
num_calibration_batches = len(test_X)
evaluate(model, test_X, test_y, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done', model)

# Convert to quantized model
torch.ao.quantization.convert(model, inplace=True)
# You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
# This warning occurs because not all modules are run in each model runs, so some
# modules may not be calibrated.
print('Post Training Quantization: Convert done', model)


# num_eval_batches = 1000
# top1, top5 = evaluate(net, criterion, testloader, neval_batches=num_eval_batches)
# print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches, top1.avg))
# compute_accuracy(net, testloader)

print(
    f'\nEvaluating accuracy on the same {len(test_X)} samples used in the JSON file...')
json_model_top1 = evaluate(model, test_X, test_y, neval_batches=len(test_X))
print(
    f'Model evaluation accuracy on {len(test_X)} JSON samples: {json_model_top1.avg:.2f}%')

# 2. Compute accuracy directly from the JSON file
input_path = args.export / "input.json"
with open(input_path, 'r') as f:
    json_data = json.load(f)

# Calculate accuracy from the saved outputs in JSON
correct_json = 0
total_json = len(json_data["pytorch_output"])
for i in range(total_json):
    # Get prediction from pytorch output in JSON
    json_output = json_data["pytorch_output"][i]
    _, predicted_json = torch.max(torch.tensor(json_output), 0)

    # Get ground truth from one-hot encoded output
    true_label = json_data["output_data"][i].index(1.0)

    if predicted_json.item() == true_label:
        correct_json += 1

json_accuracy = 100 * correct_json / total_json
print(f'JSON file accuracy on {total_json} samples: {json_accuracy:.2f}%')

# Compare the two
print(f'\nComparison: Model evaluation vs JSON file accuracy')
print(
    f'Model evaluation: {json_model_top1.avg:.2f}% | JSON file: {json_accuracy:.2f}%')

# Existing accuracy computation on full test set
accuracy = evaluate_accuracy(model, test_X, test_y, actual_samples)
print(f"{accuracy:.2f}%")

print_model_io(model, sample_input)
