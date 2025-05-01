import torch
import pathlib
from torch import nn



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.layer = nn.Softmax(dim=1)

    def forward(self, x):
        return self.layer(x)
    
path = pathlib.Path(__file__).parent.resolve().joinpath("softmax.onnx")
dummy_input = torch.randn(2, 3, 4)
model = MyModel()
torch.onnx.export(model, dummy_input, f=path,)