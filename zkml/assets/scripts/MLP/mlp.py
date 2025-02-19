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

# Load the iris data
iris = load_iris()
dataset = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target'])
print("Loaded iris data")


class Model(nn.Module):
    def __init__(self, num_hidden, layer_width):
        super(Model, self).__init__()

        layers = []
        input_size = 4  # Input features (Iris dataset has 4 features)
        output_size = 3  # Number of classes in Iris dataset

        # First hidden layer
        layers.append(nn.Linear(input_size, layer_width))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(layer_width, output_size))
        layers.append(nn.ReLU())

        # Combine layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


model = Model(num_hidden=1, layer_width=20)

train_X, test_X, train_y, test_y = train_test_split(
    dataset[dataset.columns[0:4]].values,  # use columns 0-4 as X
    dataset.target,  # use target as y
    test_size=0.2  # use 20% of data for testing
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

model_path = os.path.join('model-mlp.onnx')
data_path = os.path.join('input.json')

x = test_X[0].reshape(1, 4)
model.eval()
torch.onnx.export(model,
                  x,
                  model_path,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

data_array = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data=[data_array])
json.dump(data, open(data_path, 'w'))
