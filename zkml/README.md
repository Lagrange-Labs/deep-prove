# 🔍 ZKML Inference Proving: Deep Dive

**Welcome back to ZKML!** This document will guide you through the inner workings of ZKML, how to install it, and how to make the most of its capabilities.

## 🌐 Overview

ZKML is a framework for proving inference of neural networks using cryptographic techniques based on sumchecks and logup GKR. Thanks to these techniques, the proving time is sublinear in the size of the model, providing significant speedups compared to other inference frameworks.

The framework currently supports proving inference for both Multi-Layer Perceptron (MLP) models and Convolutional Neural Networks (CNN). It supports dense layers, ReLU, maxpool, and convolutions. The framework requantizes the output after each layer into a fixed zero-centered range; by default, we use a [-128;127] quantization range.

## ⚙️ How It Works

ZKML is built on cryptographic techniques that allow for efficient proof generation and verification of neural network inferences. It supports various layers like dense, ReLU, maxpool, and convolution, with a focus on speed and accuracy.

## 🛠️ Installation

To get started with ZKML, ensure you have the following prerequisites:

- **Python 3.6 or higher**
- **Rust and Cargo** (latest stable)
- **EZKL** (optional, for comparisons)

### Installing Rust and Cargo

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Python Dependencies

```bash
pip install -r assets/scripts/requirements.txt
```

## 🏃‍♂️ Customizing and Running Benchmarks

ZKML allows you to customize Python scripts located in `zkml/assets/scripts/` to suit your needs. Once set up, you can run the `bench.py` script to test different configurations and measure performance.

### Example Command

```bash
python bench.py --configs 5,100:7,50 --repeats 5
```

## 📥 Input Requirements

ZKML expects an ONNX model and an input JSON file to perform inference proving. Customize these inputs to match your specific model and data requirements.

## 🎈 Try It Out

Feel free to tweak the Python scripts and run `bench.py` to explore the full potential of ZKML. Whether you're optimizing for speed or accuracy, ZKML provides the tools you need to succeed.

Stay curious and keep experimenting! 🌟

## 🛤️ Status & Roadmap

This is a research-driven project and the codebase is improving at a fast pace. Here is the current status of the project:

**Features**:

[x] Prove inference of Dense layers  
[x] Prove inference of ReLU  
[x] Prove inference of MaxPool  
[x] Prove inference of Convolution  
[ ] Add support for more layer types (BatchNorm, Dropout, etc)

**Accuracy**:
[x] Layer-wise requantization (a single scaling factor per layer)  
[ ] Allowing BIT_LEN to grow without losing performance (lookup related)  
[ ] Add support for row-wise quantization for each layer to provide better accuracy

**Performance**:
[ ] Better lookup usage with more small tables  
[ ] Implement simpler GKR for logup - no need to have a full generic GKR  
[ ] Improved parallelism for logup, GKR, sumchecks  
[ ] GPU support

## 🛠️ Troubleshooting

- **CPU Affinity Issues**: Thread limiting via CPU affinity is not supported on macOS. The script will proceed without restrictions and display a warning.
- **EZKL Not Found**: When using `--run-ezkl`, ensure EZKL is properly installed and in your PATH.
- **Memory Limitations**: Large models may require substantial memory. Consider reducing model size or using a machine with more RAM if you encounter memory errors.
- **Performance Variability**: For the most consistent results, close other resource-intensive applications when running benchmarks and consider using the `--num-threads` option.

## 📄 LICENSE

This project is licensed under the [LICENSE](LICENSE) file.

## 🙏 Acknowledgements

This project is built on top of the work from scroll-tech/ceno - it re-uses the sumcheck and GKR implementation from the codebase at https://github.com/scroll-tech/ceno
