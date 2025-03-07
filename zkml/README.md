# ZKML Inference Proving

**WARNING**: **This codebase is not audited and not production ready and is provided as is. Use at your own risk.**

## Overview

ZKML is a framework for proving inference of neural networks using cryptographic techniques based on sumchecks, and logup GKR mostly. Thanks to these techniques, the proving time is actually sublinear in the size of the model and is able to provide an order of magnitude speedups compared to other inference frameworks.

The framework currently supports proving inference for both Multi-Layer Perceptron (MLP) models and Convolutional Neural Networks (CNN). Namely, it supports dense layers, relu, maxpool and convolutions.
The framework requantizes the output after each layer into a fixed zero-centered range; by default we are using [-128;127] quantization range. 

Stay tuned for a blog post explaining the technical details of the framework !

## Status & Roadmap

This is a research driven project and the codebase is improving on a fast pace. Here is the current status of the project:

**Features**:

[x] Prove inference of Dense layers
[x] Prove inference of ReLU
[x] Prove inference of MaxPool
[x] Prove inference of Convolution
[ ] Add support for more layers types (BatchNorm, Dropout, etc)

**Accuracy**:
[x] Layer-wise requantization (a single scaling factor per layer)
[ ] Allowing BIT_LEN to grow without loosing performance (lookup related)
[ ] Add supports for row-wise quantization for each layer to provide better accuracy

**Performance**:
[ ] Better lookup usage with more small tables 
[ ] Implement simpler GKR for logup - no need to have a full generic GKR
[ ] Improved parallelism for logup, gkr, sumchecks
[ ] GPU support

## Benchmark

This repo provides bench.py, a tool measuring critical metrics including:

- Proof generation time
- Verification time
- Setup time
- Inference time
- Proof size
- Output accuracy

These benchmarks help quantify the trade-offs between different approaches and configurations, guiding decisions about which implementation best suits specific use cases.

## Prerequisites

Before running the benchmarks, ensure you have the following installed:

- Python 3.6 or higher
- [Rust and Cargo](https://www.rust-lang.org/tools/install) (latest stable)
- [EZKL](https://github.com/zkonduit/ezkl#installation) (required only if running EZKL comparisons)

### Installing Rust and Cargo

If you don't have Rust and Cargo installed, follow these instructions:

```bash
# For most platforms, use rustup:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the on-screen instructions to complete the installation
# Then reload your shell:
source $HOME/.cargo/env
```

Visit the [official Rust installation page](https://www.rust-lang.org/tools/install) for platform-specific instructions.

### Installing EZKL

If you plan to run EZKL comparisons, refer to the [EZKL documentation](https://github.com/zkonduit/ezkl#installation).

### Python Dependencies

Install the required Python packages:

```bash
pip install -r assets/scripts/requirements.txt
```
**NOTE**: If you prefer to use a separated environment, you can do first:
```bash
python -m venv venv
source venv/bin/activate
```

## Building ZKML

Build the ZKML project using Cargo:

```bash
cargo build --release
```

## Running Benchmarks

The main benchmark script is `bench.py`, which supports various configurations and options.

### Basic Usage

```bash
python bench.py [options]
```

### Available Options

- `--configs`: Model configurations to test, specified as 'dense_layers,width:dense_layers,width:...'
  - Default: "1,10:2,20"
  - Example: "5,50:6,100" (benchmarks a 5-layer MLP with width 50 and a 6-layer MLP with width 100)

- `--repeats`: Number of times to repeat each benchmark for statistical significance
  - Default: 3
  - Example: `--repeats 10`

- `--output-dir`: Directory to store benchmark results
  - Default: "./bench/"
  - Example: `--output-dir my_results/`

- `--verbose`: Enable detailed output during execution
  - Example: `--verbose`

- `--run-ezkl`: Run equivalent benchmarks using EZKL for comparison
  - Example: `--run-ezkl`

- `--skip-ezkl-calibration`: Skip the EZKL calibration step (faster but potentially less accurate)
  - Example: `--skip-ezkl-calibration`

- `--ezkl-check-mode`: Set EZKL check mode ('safe' or 'unsafe')
  - Default: "safe"
  - Example: `--ezkl-check-mode unsafe`

- `--num-threads`: Limit CPU thread usage (useful for consistent results or resource management)
  - Example: `--num-threads 4`

- `--samples`: Number of input/output samples to process for both ZKML and EZKL
  - Default: 30
  - Example: `--samples 10`

- `--model-type`: Type of model to benchmark: 'mlp' for Multi-Layer Perceptron or 'cnn' for Convolutional Neural Network
  - Default: "mlp"
  - Example: `--model-type cnn`

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

Comparative benchmarking with EZKL, skipping calibration:
```bash
python bench.py --configs 4,128 --run-ezkl --skip-ezkl-calibration --verbose
```

Resource-constrained benchmarking with fewer samples:
```bash
python bench.py --num-threads 8 --configs 5,64 --samples 10
```

## Understanding Results

Benchmark results are saved as CSV files in the specified output directory:

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

## Troubleshooting

- **CPU Affinity Issues**: Thread limiting via CPU affinity is not supported on macOS. The script will proceed without restrictions and display a warning.

- **EZKL Not Found**: When using `--run-ezkl`, ensure EZKL is properly installed and in your PATH.

- **Memory Limitations**: Large models may require substantial memory. Consider reducing model size or using a machine with more RAM if you encounter memory errors.

- **Performance Variability**: For the most consistent results, close other resource-intensive applications when running benchmarks and consider using the `--num-threads` option.

## Acknowledgements

This project is built on top of the work from scroll-tech/ceno - it re-uses the sumcheck and gkr implementation
from the codebase at https://github.com/scroll-tech/ceno