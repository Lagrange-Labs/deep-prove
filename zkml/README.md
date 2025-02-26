# ZKML Benchmarking Tool

## Overview

ZKML (Zero-Knowledge Machine Learning) combines the power of machine learning with the privacy guarantees of zero-knowledge proofs. This benchmarking tool allows researchers and developers to evaluate and compare the performance of different ZKML implementations—specifically comparing our custom ZKML implementation against EZKL (a popular zero-knowledge ML framework).

The tool focuses on benchmarking Multi-Layer Perceptron (MLP) models with various configurations, measuring critical metrics including:

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

If you plan to run EZKL comparisons, install EZKL using:

```bash
cargo install ezkl
```

For detailed installation instructions or troubleshooting, refer to the [EZKL documentation](https://github.com/zkonduit/ezkl#installation).

### Python Dependencies

Install the required Python packages:

```bash
pip install -r assets/scripts/MLP/requirements.txt
```

Key dependencies include:
- PyTorch (for model creation)
- NumPy
- Psutil (for processor management)
- Matplotlib (for visualizing results)

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
  - Default: "3,4:4,8"
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

- `--num-threads`: Limit CPU thread usage (useful for consistent results or resource management)
  - Example: `--num-threads 4`

### Example Commands

Simple benchmark with default settings:
```bash
python bench.py
```

Benchmark specific model configurations with multiple runs:
```bash
python bench.py --configs 5,100:7,50 --repeats 5
```

Comparative benchmarking with EZKL:
```bash
python bench.py --configs 4,128 --run-ezkl --verbose
```

Resource-constrained benchmarking:
```bash
python bench.py --num-threads 8 --configs 5,64
```

## Understanding Results

Benchmark results are saved as CSV files in the specified output directory:

- `zkml_d{dense}_w{width}.csv`: ZKML benchmark results
- `ezkl_d{dense}_w{width}.csv`: EZKL benchmark results (if enabled)

Each CSV file contains these performance metrics for each run:
- Setup time (ms)
- Inference time (ms)
- Proving time (ms)
- Verification time (ms)
- Accuracy (boolean, 1=correct, 0=incorrect)
- Proof size (KB)

The script also prints the average accuracy for each configuration after completion.

## Troubleshooting

- **CPU Affinity Issues**: Thread limiting via CPU affinity is not supported on macOS. The script will proceed without restrictions and display a warning.

- **EZKL Not Found**: When using `--run-ezkl`, ensure EZKL is properly installed and in your PATH.

- **Memory Limitations**: Large models may require substantial memory. Consider reducing model size or using a machine with more RAM if you encounter memory errors.

- **Performance Variability**: For the most consistent results, close other resource-intensive applications when running benchmarks and consider using the `--num-threads` option.