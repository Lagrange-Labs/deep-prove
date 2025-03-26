# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful

Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
ImageNet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: 'airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck'. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalize the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Load and normalize CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it's extremely easy to load CIFAR10.
"""
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os
import json
import argparse
from pathlib import Path

# Add argument parser similar to mlp.py
parser = argparse.ArgumentParser(description="CIFAR10 CNN with ONNX export")
parser.add_argument("--export", type=Path, default=Path('bench'), 
                    help="Directory to export the model to (default: bench)")
parser.add_argument("--num-samples", type=int, default=100, 
                    help="Number of test samples to export")
parser.add_argument("--num-params", type=int, default=None,
                    help="Target number of parameters for the model (default: None, uses default model)")
parser.add_argument("--distribution", action="store_true",
                    help="Show distribution of model weights")

args = parser.parse_args()

# Ensure the export folder exists
if not args.export.exists() or not args.export.is_dir():
    print(f"❌ Error: export folder '{args.export}' does not exist or is not a directory.")
    exit(1)

# Add after parser section
print(f"\n🔍 Running CIFAR10 CNN with export path: {args.export}")
print(f"📊 Will export {args.num_samples} samples as input/output pairs\n")

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

# Add before transform
print("📦 Preparing data transformations...")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# Add before trainset
print("🔽 Loading and preparing training dataset...")

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Add after classes definition
print(f"✅ Datasets loaded successfully")
print(f"👉 CIFAR10 has {len(trainset)} training images and {len(testset)} test images")
print(f"👉 {len(classes)} classes: {', '.join(classes)}\n")

########################################################################
# Let us show some of the training images, for fun.


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# Add before getting training images
print("📊 Fetching sample training images...")

dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch
import torch.nn as nn
import torch.nn.functional as F

def estimate_params(c1, c2, fc1, fc2, fc3):
    conv1_params = (3 * 5 * 5 + 1) * c1
    conv2_params = (c1 * 5 * 5 + 1) * c2
    fc1_params = (c2 * 5 * 5 + 1) * fc1
    fc2_params = (fc1 + 1) * fc2
    fc3_params = (fc2 + 1) * fc3
    return conv1_params + conv2_params + fc1_params + fc2_params + fc3_params

class Net(nn.Module):
    def __init__(self, target_params=None):
        super().__init__()
        
        # Default values
        c1, c2 = 6, 16
        fc1, fc2, fc3 = 120, 84, 10
        
        if target_params:
            # Adjust parameters iteratively to fit within target
            scale = (target_params / estimate_params(c1, c2, fc1, fc2, fc3)) ** 0.5
            c1, c2 = int(c1 * scale), int(c2 * scale)
            fc1, fc2 = int(fc1 * scale), int(fc2 * scale)
            
            # Recalculate to get final parameter count
            final_params = estimate_params(c1, c2, fc1, fc2, fc3)
            assert abs(final_params - target_params) / target_params <= 0.05, "Final params exceed 5% tolerance"
        
        self.conv1 = nn.Conv2d(3, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.fc1 = nn.Linear(c2 * 5 * 5, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#class Net(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 10)
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = torch.flatten(x, 1)  # flatten all dimensions except batch
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x


if args.num_params:
    print(f"🏗️ Initializing neural network with target parameter count: {args.num_params:,}...")
    try:
        net = Net(target_params=args.num_params)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        print(f"   Conv1: {net.conv1.out_channels} channels")
        print(f"   Conv2: {net.conv2.out_channels} channels")
        print(f"   FC1: {net.fc1.out_features} features")
        print(f"   FC2: {net.fc2.out_features} features")
    except AssertionError as e:
        print(f"❌ Error: {e}")
        print(f"Using default model instead.")
        net = Net()  # Use default parameters
else:
    print("🏗️ Initializing default neural network...")
    net = Net()  # Use the default parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"✅ Default model created with {total_params:,} parameters")

# Add after class definition
print("🏗️ Initializing neural network...")

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

# Add before training loop
print("\n🚀 Starting model training...")
print(f"🔄 Training for 2 epochs with batch size {batch_size}")

for epoch in range(2):  # loop over the dataset multiple times
    print(f"\n💫 Epoch {epoch+1}/2")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print more frequently
            print(f'    Batch {i+1:5d} - Loss: {running_loss / 500:.4f}')
            running_loss = 0.0

print('Finished Training')

# Add before saving model
print("\n💾 Training complete! Saving model...")

########################################################################
# Let's quickly save our trained model:

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################

net.eval()
dummy_input = torch.randn(1, 3, 32, 32)
model_path = args.export / "model.onnx"
torch.onnx.export(net, dummy_input, model_path, export_params=True, opset_version=12,
                  do_constant_folding=True, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}})

print(f"Model onnx exported to {model_path}")

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(f"Running on {device}")

########################################################################
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# .. code:: python
#
#         inputs, labels = data[0].to(device), data[1].to(device)
#
# Why don't I notice MASSIVE speedup compared to CPU? Because your network
# is really small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/

# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
del dataiter
# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%

# Prepare data arrays for JSON export
input_data = []
output_data = []
pytorch_output = []

# Process a subset of test samples
num_samples = min(args.num_samples, len(testset))
sample_indices = np.random.choice(len(testset), num_samples, replace=False)

# Process each selected test sample
for i in sample_indices:
    # Get test image and label
    test_x, true_label = testset[i]
    
    # Create a batch dimension for the model
    test_x_batch = test_x.unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        model_output = net(test_x_batch).squeeze(0)
    
    # Convert input tensor to list
    input_list = test_x.detach().numpy().reshape([-1]).tolist()
    
    # Create one-hot encoded ground truth output
    one_hot_output = [0.0] * 10  # For 10 classes in CIFAR10
    one_hot_output[true_label] = 1.0
    
    # Store the data
    input_data.append(input_list)
    output_data.append(one_hot_output)
    pytorch_output.append(model_output.tolist())

# Save multiple input/output pairs to JSON
data = {
    "input_data": input_data, 
    "output_data": output_data, 
    "pytorch_output": pytorch_output
}

input_path = args.export / "input.json"
json.dump(data, open(input_path, 'w'), indent=2)
print(f"Input/Output data for {num_samples} samples exported to {input_path}")

def plot_weight_distribution(model):
    """Plot the distribution of weights across all layers (conv and linear)."""
    # Collect all weights from conv and linear layers
    all_weights = []
    layer_names = []
    
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            # Flatten weights and add to list
            weights = layer.weight.data.cpu().numpy().flatten()
            all_weights.append(weights)
            layer_names.append(f"{name}")
    
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

# Generate weight distribution plot if requested
if args.distribution:
    print("Generating weight distribution plot...")
    plot_weight_distribution(net)
    print("Weight distribution plot saved as 'weight_distribution.png'")