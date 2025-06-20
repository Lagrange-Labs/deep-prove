# ZKML Inference Proving

**WARNING**: **This codebase is not audited and not production ready and is provided as is. Use at your own risk.**

**Welcome back to Deep Prove framework!** This document will guide you through the inner workings of ZKML, how to install it, and how to make the most of its capabilities.

## 🌐 Overview

DeepProve is a framework for proving inference of neural networks using cryptographic techniques based on sumchecks, and logup GKR mostly. Thanks to these techniques, the proving time is actually sublinear in the size of the model and is able to provide an order of magnitude speedups compared to other inference frameworks.

The framework currently supports proving inference for both Multi-Layer Perceptron (MLP) models and Convolutional Neural Networks (CNN). Namely, it supports dense layers, relu, maxpool and convolutions.
The framework requantizes the output after each layer into a fixed zero-centered range; by default we are using [-128;127] quantization range. 

Stay tuned for a blog post explaining the technical details of the framework !

## Status & Roadmap

This is a research driven project and the codebase is improving on a fast pace. Here is the current status of the project:

**Features**:

- [x] Prove inference of Dense layers
- [x] Prove inference of ReLU
- [x] Prove inference of MaxPool
- [x] Prove inference of Convolution
- [ ] Add support for more layers types (BatchNorm, Dropout, etc)

**Accuracy**:
- [x] Layer-wise requantization (a single scaling factor per layer)
- [ ] Allowing BIT_LEN to grow without loosing performance (lookup related)
- [ ] Add supports for row-wise quantization for each layer to provide better accuracy

**Performance**:
- [ ] Better lookup usage with more small tables
- [ ] Implement simpler GKR for logup - no need to have a full generic GKR
- [ ] Improved parallelism for logup, gkr, sumchecks
- [ ] GPU support

## Benchmark

For high level benchmark comparison, check out the main [README](https://github.com/Lagrange-Labs/deep-prove) !

To run your own benchmark, this repo provides bench.py that can run deep prove framework against a generated onnx file and an input file. 
Check out the folder `assets/scripts/` to see the PyTorch scripts generating the CNN and MLP model used in the benchmark !


## ⚙️ How It Works

ZKML is built on cryptographic techniques that allow for efficient proof generation and verification of neural network inferences. It supports various layers like dense, ReLU, maxpool, and convolution, with a focus on speed and accuracy.

## 🛠️ Installation

To get started with ZKML, ensure you have the following prerequisites:

- **Python 3.10 or higher**
- **Rust and Cargo** (latest stable)
- **EZKL** (optional, for comparisons)

### Installing Rust and Cargo

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Visit the [official Rust installation page](https://www.rust-lang.org/tools/install) for platform-specific instructions.

### Installing EZKL

If you plan to run EZKL comparisons, refer to the [EZKL documentation](https://github.com/zkonduit/ezkl#installation).

### Python Dependencies

```bash
pip install -r assets/scripts/requirements.txt
```
**NOTE**: If you prefer to use a separated environment, you can do first:
```bash
python -m venv venv
source venv/bin/activate
```

ZKML allows you to customize Python scripts located in `zkml/assets/scripts/` to suit your needs. Once set up, you can run the `bench.py` script to test different configurations and measure performance.

### Example Command

```bash
cargo build --release
```

## Running Benchmarks

The main benchmark script is `bench.py`, which supports various configurations and options.

### Basic Usage

```bash
python bench.py [options]
```

### Example Commands

Simple benchmark with default settings:
```bash
python bench.py
```

Benchmark specific MLP model configurations with multiple runs:
```bash
python bench.py --configs 5,100:7,50 --repeats 5
```

Benchmark a CNN model:
```bash
python bench.py --model-type cnn
```

Comparative benchmarking with EZKL:
```bash
python bench.py --configs 4,128 --run-ezkl --verbose
```

Resource-constrained benchmarking with fewer samples:
```bash
python bench.py --num-threads 8 --configs 5,64 --samples 10
```

## Breakdown of deeo prove performance

Add a env var before running the script where you will get a CSV file where each line is one invocation of a function. Aggregating these lines on a spreadsheet engine will give you a good idea where the bottlenecks are:
```bash
TIMED_OUTPUT=prover_perf.csv python bench.py ...
```

## 🎈 Try It Out

Feel free to tweak the Python scripts and run `bench.py` to explore the full potential of ZKML. 
This is a research-driven project and the codebase is improving at a fast pace. Here is the current status of the project:
- `zkml_d{dense}_w{width}.csv`: ZKML benchmark results for MLP models
- `zkml_cnn.csv`: ZKML benchmark results for CNN models
- `ezkl_d{dense}_w{width}.csv`: EZKL benchmark results (if enabled) for MLP models
- `ezkl_cnn.csv`: EZKL benchmark results (if enabled) for CNN models

Each CSV file contains these performance metrics for each run:
- Setup time (ms)
- Inference time (ms) 
- Proving time (ms)
- EZKL full proving time (ms) - Total time for witness generation + proving in EZKL
- Verification time (ms)
- Accuracy (boolean, 1=correct, 0=incorrect)
- Proof size (KB)

The script also generates a summary table showing averages across all runs for each configuration.

Stay curious and keep experimenting! 🌟

### Documentation

We use [mdBook](https://github.com/rust-lang/mdBook) together with a [katex extension](https://github.com/lzanini/mdbook-katex) to produce technical documentation related to DeepProve, if you already have Cargo and Rust installed you can simply run
```bash
cargo install mdbook
```
followed  by 
```bash
cargo install mdbook-katex
```
to install it, for more info on installing mdBook check out their guide [here](https://rust-lang.github.io/mdBook/guide/installation.html).

To build and open the docs, from the `zkml` directory run: 
```bash 
mdbook build docs --open
```
This will open a copy of the docs in your default web browser.


## Caveats on EZKL comparison

**Unfair comparison**: The ezkl binary does not allow to configure the exact amount of logrows used for proving.
Therefore, it uses the fastest possible way to prove but incurring a MUCH larger time to verify.
In production scenario, the proving time should be way higher with a much smaller verification time.

The TLDR is that it is hard to get a fair apple-to-apple comparison between ZKML and EZKL.

**Invalid Verification**: 
* When not using the calibration step, EZKL verification actually fails, and proving time is much higher.
* Calibration makes verification succeed but force to trim the KZG params at proving time, which takes a lot of time.
So both time, proving + trimming and proving alone are included in the benchmark since both are offering different trade-offs that don't seem to be customizable from the CLI.

**GPU Code**: By default on MacOS with Apple Silicon, the ezkl binary is compiled with GPU support. We currently do not  
support GPU code. Installing the CPU version from the source code is the only way to go. 

## 🛠️ Troubleshooting

- **CPU Affinity Issues**: Thread limiting via CPU affinity is not supported on macOS. The script will proceed without restrictions and display a warning.
- **EZKL Not Found**: When using `--run-ezkl`, ensure EZKL is properly installed and in your PATH.
- **Memory Limitations**: Large models may require substantial memory. Consider reducing model size or using a machine with more RAM if you encounter memory errors.
- **Performance Variability**: For the most consistent results, close other resource-intensive applications when running benchmarks and consider using the `--num-threads` option.

## 📄 LICENSE

This project is licensed under the [LICENSE](LICENSE) file.

## 🙏 Acknowledgements

This project is built on top of the work from scroll-tech/ceno - it re-uses the sumcheck and GKR implementation from the codebase at https://github.com/scroll-tech/ceno
