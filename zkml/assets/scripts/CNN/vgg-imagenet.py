from torchvision import transforms
from PIL import Image
import urllib
import torch

available_vgg = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']


def gen_vgg_onnx(vgg_cfg):
    model = torch.hub.load('pytorch/vision:v0.10.0', vgg_cfg, pretrained=True)
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
    onnx_file_path = f"{vgg_cfg}.onnx"

    # Export the model to ONNX format
    torch.onnx.export(
        model,                  # Model to export
        input_batch,            # Sample input tensor
        onnx_file_path,         # Output file path
        export_params=True,     # Store trained parameter weights
        opset_version=13,       # ONNX opset version (11 is widely compatible)
        do_constant_folding=True,  # Optimize by folding constants
        # input_names=['input'],  # Name of the input tensor
        # output_names=['output']  # Name of the output tensor
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
