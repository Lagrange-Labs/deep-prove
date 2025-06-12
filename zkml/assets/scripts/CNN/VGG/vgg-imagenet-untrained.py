from functools import partial
from typing import Any, cast, Dict, List, Optional, Union
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import sys
import urllib


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, weights) -> VGG:
    if weights is not None:
        print("Cannot handle actual weights yet")
        sys.exit(0)

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))

    return model


def vgg11(weights) -> VGG:

    return _vgg("A", False, weights)


def vgg11_bn(weights) -> VGG:

    return _vgg("A", True, weights)


def vgg13(weights) -> VGG:

    return _vgg("B", False, weights)


def vgg13_bn(weights) -> VGG:

    return _vgg("B", True, weights)


def vgg16(weights) -> VGG:

    return _vgg("C", False, weights)


def vgg16_bn(weights) -> VGG:

    return _vgg("C", True, weights)


def vgg19(weights) -> VGG:

    return _vgg("D", False, weights)


def vgg19_bn(weights) -> VGG:

    return _vgg("D", True, weights)


available_vgg = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

# Map VGG config names to their model functions and weights
vgg_model_map = {
    'vgg11': (vgg11, None),
    'vgg11_bn': (vgg11_bn, None),
    'vgg13': (vgg13, None),
    'vgg13_bn': (vgg13_bn, None),
    'vgg16': (vgg16, None),
    'vgg16_bn': (vgg16_bn, None),
    'vgg19': (vgg19, None),
    'vgg19_bn': (vgg19_bn, None),
}


def gen_vgg_onnx(vgg_cfg):
    if vgg_cfg not in vgg_model_map:
        raise ValueError(f"Unsupported VGG config: {vgg_cfg}")

    # Get the model function and weights
    model_fn, weights = vgg_model_map[vgg_cfg]

    # Load the model with weights
    model = model_fn(weights=weights)
    model.eval()

    """All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
    The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
    and `std = [0.229, 0.224, 0.225]`.
    """

    # Download an example image from the pytorch website
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes

    # print(output[0])

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)

    # Define the output ONNX file name
    onnx_file_path = f"{vgg_cfg}-untrained.onnx"

    # Export the model to ONNX format
    torch.onnx.export(
        model,                  # Model to export
        input_batch,            # Sample input tensor
        onnx_file_path,         # Output file path
        export_params=True,     # Store trained parameter weights
        opset_version=13,       # ONNX opset version (11 is widely compatible)
        do_constant_folding=True,  # Optimize by folding constants
        input_names=['input'],  # Name of the input tensor
        output_names=['output']  # Name of the output tensor
    )

    print(f"Model has been saved as {onnx_file_path}")

    # Download an example image from the pytorch website
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    gen_vgg_onnx("vgg11")
    gen_vgg_onnx("vgg11_bn")
