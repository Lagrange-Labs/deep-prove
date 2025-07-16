#!/usr/bin/env python
import subprocess
import os
import time
import sys
from pathlib import Path
import argparse
import shutil
import csv
import logging
from typing import Callable, Dict, List, Union
import re
import json
import psutil  # Import psutil for CPU affinity
import platform
import pandas as pd
import tabulate

logging.basicConfig(level=logging.INFO)

# Constants for CSV column headers
CONFIG = "config"
RUN = "run"
SETUP = "setup (ms)"
INFERENCE = "inference (ms)"
PROVING = "proving (ms)"
VERIFYING = "verifying (ms)"
ACCURACY = "accuracy (bool)"
PROOF_SIZE = "proof size (KB)"
SAMPLE = "sample"
EZKL_FULL_PROVING = "ezkl_full_proving (ms)"  # New constant for full proving time

# Constants for file paths
MODEL = "model.onnx"
INPUT = "input.json"
EZKL_KZG_PARAMS = "kzg.params"


class CSVBencher:

    def __init__(self, headers: List[str]):
        self.headers = headers
        self.data: Dict[str, str] = {}

    def _check(self, column: str):
        if column in self.data:
            raise ValueError("CSVBencher only handles one row for now")
        if column not in self.headers:
            raise ValueError(f"Column {column} does not exist")

    def r(self, column: str, task: Union[Callable, List[str]]):
        """Runs a function or a shell command, measures execution time, and stores the result."""
        self._check(column)
        start_time = time.perf_counter()

        if isinstance(task, list):
            result = subprocess.run(task, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Command {task} failed with exit code {result.returncode}: {result}"
                )
        else:
            result = task()

        elapsed = int(
            (time.perf_counter() - start_time) * 1000)  # Convert to ms
        logging.info(f"STEP: {column} took {elapsed}ms")
        self.data[column] = str(elapsed)
        return result

    def set(self, column: str, value: str):
        """Sets a value for a column after validation."""
        self._check(column)
        self.data[column] = str(value)

    def flush(self, fname: str):
        """Writes the collected data to a CSV file."""
        file_path = Path(fname)
        file_exists = file_path.exists()

        with file_path.open("a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(self.headers)  # Write headers if file is new

            values = [self.data.get(header, "") for header in self.headers]
            writer.writerow(values)

        logging.info(f"Flushed data to {fname}")


def ensure_command_exists(command):
    if shutil.which(command) is None:
        print(f"❌ Error: '{command}' is not installed or not in PATH.",
              file=sys.stderr)
        sys.exit(1)  # Exit with an error code


def ex(cmd, verbose=False):
    print(f"\n🔄 Running command: {' '.join(cmd)}")
    start_time = time.perf_counter()
    # Pass through all environment variables
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    end_time = time.perf_counter()

    elapsed_time_ms = (end_time - start_time) * 1000

    if verbose and result.stdout:
        print(f"📝 Output:\n{result.stdout}")

    if result.returncode != 0:
        print(f"❌ Error (code {result.returncode}):\n{result.stderr}",
              file=sys.stderr)
        sys.exit(result.returncode)

    print(f"⏱️  Took {elapsed_time_ms:.0f}ms")

    return {"stdout": result.stdout, "elapsed_time_ms": elapsed_time_ms}


# Default paths
DEFAULT_BENCH_FOLDER = Path("./bench/")
PYTORCH_SCRIPT = "./assets/scripts/MLP/mlp.py"


def create_model(num_dense,
                 layer_width,
                 output_dir,
                 verbose,
                 num_samples,
                 model_type,
                 param_cnn=None):
    """Create a model with the specified parameters"""
    print(
        f"\n⚡ Creating {model_type.upper()} model with {num_dense} dense layers, each with width {layer_width}"
    )

    # Create the model script path based on model type
    match model_type:
        case "mlp":
            script_path = Path("assets/scripts/MLP/mlp.py")
            cmd = [
                "python",
                str(script_path), "--num-dense",
                str(num_dense), "--layer-width",
                str(layer_width), "--export",
                str(output_dir), "--num-samples",
                str(num_samples)
            ]
        case "covid":
            script_path = Path("assets/scripts/covid/covid.py")
            cmd = [
                "python",
                str(script_path),
                "--export",
                str(output_dir),
            ]
        case _:
            script_path = Path("assets/scripts/CNN/cifar-cnn.py")
            cmd = [
                "python",
                str(script_path), "--export",
                str(output_dir), "--num-samples",
                str(num_samples)
            ]
            # Add parameter count for CNN if specified
            if param_cnn is not None:
                cmd.extend(["--num-params", str(param_cnn)])
                script_path = Path("assets/scripts/CNN/cifar-cnn.py")

    # Ensure the script exists
    if not script_path.exists():
        print(f"❌ Error: Model script not found at {script_path}")
        sys.exit(1)

    if verbose:
        print(f"📋 Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error creating model:\n{result.stderr}")
        sys.exit(1)

    if verbose:
        print(result.stdout)

    # Return paths to model and input/output file
    model_path = output_dir / MODEL
    input_path = output_dir / INPUT

    if not model_path.exists() or not input_path.exists():
        missing = []
        if not model_path.exists():
            missing.append(str(model_path))
        if not input_path.exists():
            missing.append(str(input_path))
        print(
            f"❌ Error: The following files were not created: {', '.join(missing)}"
        )
        sys.exit(1)

    print(f"✅ Model created successfully: {model_path}")
    print(f"✅ Input file created: {input_path}")

    return model_path, input_path


def run_zkml_benchmark(config_name,
                       output_dir,
                       verbose,
                       num_samples,
                       skip_proving=False):
    """Run ZKML benchmark and save results to CSV"""
    zkml_csv = output_dir / f"zkml_{config_name}.csv"

    print(f"Running ZKML benchmark for {config_name}")
    ensure_command_exists("cargo")

    # Build the command with optional skip-proving flag
    cmd = [
        "cargo", "run", "--release", "--bin", "bench",
        "--",
        "-i", str(output_dir / INPUT),
        "-o", str(output_dir / MODEL),
        "--bench", str(zkml_csv),
        "--num-samples", str(num_samples)
    ]

    if skip_proving:
        cmd.append("--skip-proving")

    _ = ex(cmd, verbose=verbose)
    print("ZKML benchmark completed")
    return zkml_csv


def run_ezkl_benchmark(config_name,
                       run_index,
                       output_dir,
                       verbose,
                       num_samples,
                       skip_calibration=False,
                       check_mode="safe",
                       show_accuracy=False):
    """Run EZKL benchmark for each input/output pair and save results to CSV"""
    # Create absolute paths before changing directory
    ezkl_csv = output_dir / f"ezkl_{config_name}.csv"
    absolute_ezkl_csv = ezkl_csv.absolute()

    print(f"Running EZKL benchmark for {config_name}, run {run_index}")
    print(f"Moving to {output_dir} for ezkl bench")
    original_dir = os.getcwd()
    os.chdir(output_dir)
    ensure_command_exists("ezkl")

    try:
        LOGROWS = 24

        # Load the original input JSON file to get all input/output pairs
        with open(INPUT, "r") as f:
            original_input_data = json.load(f)

        # Check how many input/output pairs we have
        available_samples = len(original_input_data["input_data"])

        # Apply the sample limit
        max_samples = min(available_samples, num_samples)
        print(
            f"Found {available_samples} input/output pairs in {INPUT}, will process {max_samples}"
        )

        # Run setup steps once (these don't depend on the input data)
        ex([
            "ezkl", "gen-settings", "-K",
            str(LOGROWS), "-M", MODEL, "--check-mode", check_mode
        ],
           verbose=verbose)

        # Calibrate only if not skipping calibration
        if not skip_calibration:
            _ = ex([
                "ezkl", "calibrate-settings", "-M", MODEL, "-D", INPUT,
                "--max-logrows",
                str(LOGROWS)
            ], verbose=verbose)
        else:
            print("Skipping EZKL calibration step as requested")

        if not Path(EZKL_KZG_PARAMS).exists():
            print("Downloading SRS params")
            ex([
                "ezkl", "get-srs", "--logrows",
                str(LOGROWS), "--srs-path", EZKL_KZG_PARAMS
            ],
               verbose=verbose)
        _ = ex(["ezkl", "compile-circuit", "-M", MODEL], verbose=verbose)

        # Run setup once and measure time
        setup_result = ex(["ezkl", "setup", "--srs-path", EZKL_KZG_PARAMS],
                          verbose=verbose)
        setup_time_ms = setup_result["elapsed_time_ms"]

        # Process each input/output pair up to the limit
        for sample_idx in range(max_samples):
            print(f"\n[+] Processing EZKL sample {sample_idx+1}/{max_samples}")

            # Create a temporary JSON file with just this input/output pair
            temp_input = {
                "input_data": [original_input_data["input_data"][sample_idx]],
                "output_data":
                [original_input_data["output_data"][sample_idx]]
            }

            temp_input_file = f"temp_input_{sample_idx}.json"
            with open(temp_input_file, "w") as f:
                json.dump(temp_input, f)

            # Initialize bencher with the new column
            bencher = CSVBencher([
                CONFIG, RUN, SAMPLE, SETUP, INFERENCE, PROVING,
                EZKL_FULL_PROVING, VERIFYING, ACCURACY, PROOF_SIZE
            ])
            bencher.set(CONFIG, config_name)
            bencher.set(RUN, str(run_index))
            bencher.set(SAMPLE, str(sample_idx))
            bencher.set(SETUP, str(setup_time_ms))

            # Capture the time for gen-witness
            witness_result = ex(["ezkl", "gen-witness", "-D", temp_input_file],
                                verbose=verbose)
            witness_time_ms = witness_result["elapsed_time_ms"]
            bencher.set(INFERENCE, str(witness_time_ms))

            # We already have the witness gen time, so don't need to run it again
            # Now measure proving time
            proving_result = ex(
                ["ezkl", "prove", "--srs-path", EZKL_KZG_PARAMS],
                verbose=verbose)
            proving_time_wall = proving_result["elapsed_time_ms"]

            # Full proving time is the sum of witness generation and proving times
            full_proving_time_ms = witness_time_ms + proving_time_wall
            print(
                f"Full proving time: {witness_time_ms}ms + {proving_time_wall}ms = {full_proving_time_ms}ms"
            )
            bencher.set(EZKL_FULL_PROVING, str(full_proving_time_ms))

            # Extract the proof time using regex (same as before)
            proof_time_match = re.search(
                r"\[.*ezkl::pfsys\] - proof took (\d+\.\d+)",
                proving_result["stdout"])
            if proof_time_match:
                proof_time_seconds = float(proof_time_match.group(1))
                proof_time_ms = int(proof_time_seconds * 1000)
                print(f"Extracted proof time: {proof_time_ms}ms")
            else:
                print("Could not extract proof time, using full command time")
                proof_time_ms = proving_result["elapsed_time_ms"]

            # Sum the witness and proof times
            total_proving_time_ms = witness_time_ms + proof_time_ms
            print(
                f"Extracting proving time: {witness_time_ms}ms + {proof_time_ms}ms = {total_proving_time_ms}ms"
            )
            bencher.set(PROVING, f"{total_proving_time_ms:.2f}")

            # Run verification with error handling
            # verify_out = ex(["ezkl", "verify", "--srs-path", EZKL_KZG_PARAMS], verbose=verbose)
            start_time = time.perf_counter()
            verify_result = subprocess.run(
                ["ezkl", "verify", "--srs-path", EZKL_KZG_PARAMS],
                capture_output=True,
                text=True)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            # elapsed_ms = verify_out["elapsed_time_ms"]
            # verify_result = verify_out["stdout"]

            # Store the time regardless of success or failure
            bencher.set(VERIFYING, str(elapsed_ms))

            # Check if verification succeeded
            if verify_result.returncode != 0:
                print("\n⚠️ WARNING: EZKL verification FAILED ⚠️")
                print(f"Error message: {verify_result.stderr}")
                print("This may be expected if using --skip-ezkl-calibration")
                print(
                    "Continuing with benchmark despite verification failure...\n"
                )
            else:
                print("✅ EZKL verification succeeded")

            # Extract outputs and check accuracy
            with open("proof.json", "r") as f:
                proof_data = json.load(f)

            # Get full EZKL outputs
            full_ezkl_outputs = [
                float(x) for x in proof_data["pretty_public_inputs"]
                ["rescaled_outputs"][0]
            ]

            # Get expected PyTorch outputs
            pytorch_outputs = [float(x) for x in temp_input["output_data"][0]]
            expected_output_length = len(pytorch_outputs)

            # Truncate EZKL outputs to match expected length
            ezkl_outputs = full_ezkl_outputs[:expected_output_length]

            # Now calculate argmax on the correctly sized outputs
            ezkl_argmax = ezkl_outputs.index(max(ezkl_outputs))
            pytorch_argmax = pytorch_outputs.index(max(pytorch_outputs))

            # Only print detailed output if show_accuracy is enabled
            if show_accuracy:
                print(
                    f"EZKL outputs (full length: {len(full_ezkl_outputs)}, using first {expected_output_length})"
                )
                print(f"EZKL output: {ezkl_outputs}, argmax: {ezkl_argmax}")
                print(
                    f"PyTorch output: {pytorch_outputs}, argmax: {pytorch_argmax}"
                )

            is_correct = 1 if ezkl_argmax == pytorch_argmax else 0
            bencher.set(ACCURACY, str(is_correct))
            if show_accuracy:
                print(f"Correctness check: {'PASS' if is_correct else 'FAIL'}")
            else:
                # Just log that accuracy was computed but not shown
                print("Accuracy computed (use --show-accuracy to see details)")

            # Extract proof size in KB
            proof_size_kb = len(proof_data["proof"]) / 1024.0
            bencher.set(PROOF_SIZE, f"{proof_size_kb:.3f}")
            print(f"Proof size: {proof_size_kb:.3f} KB")

            # Write results to CSV
            bencher.flush(absolute_ezkl_csv)

            # Clean up temporary file
            os.remove(temp_input_file)

        return ezkl_csv
    finally:
        # Always return to the original directory
        os.chdir(original_dir)


def run_benchmark(num_dense,
                  layer_width,
                  run_index,
                  output_dir,
                  verbose,
                  run_ezkl,
                  num_samples,
                  model_type,
                  skip_ezkl_calibration=False,
                  ezkl_check_mode="safe",
                  show_accuracy=False,
                  skip_model_creation=False,
                  param_cnn=None,
                  skip_proving=False):
    """Run a benchmark for a specific model configuration and parameter."""
    # Create directory for this configuration if it doesn't exist
    config_name = f"d{num_dense}_w{layer_width}" if model_type == "mlp" else "cnn"

    print(f"\n{'='*80}")
    print(
        f"Running benchmark: model={model_type}, dense={num_dense}, width={layer_width}, run={run_index}"
    )
    print(f"{'='*80}\n")

    # Step 1: Create model with specified number of samples (if not skipped)
    if not skip_model_creation:
        create_model(num_dense, layer_width, output_dir, verbose, num_samples,
                     model_type, param_cnn)
    else:
        print("Skipping model creation, using existing model and input files")
        # Check if required files exist
        model_path = output_dir / MODEL
        input_path = output_dir / INPUT
        if not model_path.exists() or not input_path.exists():
            print("❌ Error: Skipping model creation but required files don't exist:")
            if not model_path.exists():
                print(f"  Missing model file: {model_path}")
            if not input_path.exists():
                print(f"  Missing input file: {input_path}")
            sys.exit(1)

    # Step 2: Run ZKML benchmark
    zkml_csv = run_zkml_benchmark(config_name, output_dir, verbose,
                                  num_samples, skip_proving)

    # Step 3: Calculate PyTorch accuracy from the input_output.json
    pytorch_csv = calculate_pytorch_accuracy(config_name, run_index,
                                             output_dir)

    # Step 4: Run EZKL benchmark if requested
    ezkl_csv = None
    if run_ezkl:
        print(f"\n📊 Running EZKL benchmark for run {run_index}...")
        ezkl_csv = run_ezkl_benchmark(config_name, run_index, output_dir,
                                      verbose, num_samples,
                                      skip_ezkl_calibration, ezkl_check_mode,
                                      show_accuracy)
        print(f"Results saved to {zkml_csv}, {pytorch_csv}, and {ezkl_csv}")
    else:
        print(
            f"Results saved to {zkml_csv} and {pytorch_csv} (EZKL comparison skipped)"
        )

    return zkml_csv, ezkl_csv, pytorch_csv


def argmax(values):
    """Return the index of the maximum value in a list."""
    return values.index(max(values)) if values else None


def calculate_pytorch_accuracy(config_name, run_index, output_dir):
    """Calculate accuracy of PyTorch model from the input/output JSON"""
    print(f"Calculating PyTorch accuracy for {config_name}")

    # Create the output CSV path
    pytorch_csv = output_dir / f"pytorch_{config_name}.csv"
    absolute_pytorch_csv = pytorch_csv.absolute()

    # Load the input/output JSON
    input_json_path = output_dir / INPUT
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Ensure we have pytorch_output in the data - panic if missing
    if "pytorch_output" not in data:
        print("❌ Error: input/output JSON does not contain PyTorch outputs")
        print("Please regenerate the input file with an updated script that includes pytorch_output.")
        sys.exit(1)

    # Calculate accuracy for each sample
    for i, (expected_output, pytorch_output) in enumerate(
            zip(data["output_data"], data["pytorch_output"])):
        # Create a new bencher for each sample
        bencher = CSVBencher([CONFIG, RUN, SAMPLE, ACCURACY])

        # Get expected class (argmax of one-hot encoded output)
        expected_class = argmax(expected_output)

        # Get PyTorch prediction (argmax of raw outputs)
        predicted_class = argmax(pytorch_output)

        # Compute accuracy (1 if match, 0 if not)
        accuracy = 1 if expected_class == predicted_class else 0

        # Add to bencher
        bencher.set(CONFIG, config_name)
        bencher.set(RUN, str(run_index))
        bencher.set(SAMPLE, str(i))
        bencher.set(ACCURACY, str(accuracy))

        # Flush to CSV
        bencher.flush(str(absolute_pytorch_csv))

    return pytorch_csv


def set_cpu_affinity(max_threads: int):
    """Set CPU affinity to limit the process and its children to `max_threads` logical CPUs."""
    num_logical_processors = psutil.cpu_count(logical=True)

    # Check if the requested number of threads is valid
    if max_threads < 1:
        print("❌ Error: The number of threads must be at least 1.")
        sys.exit(1)
    if max_threads > num_logical_processors:
        print(
            f"❌ Error: Requested {max_threads} threads, but only {num_logical_processors} logical processors are available."
        )
        sys.exit(1)

    # Check if the platform supports setting CPU affinity
    if platform.system() in ["Linux", "Windows"]:
        # Select up to `max_threads` available logical CPUs
        selected_cpus = list(range(max_threads))

        # Apply CPU affinity
        p = psutil.Process(os.getpid())
        p.cpu_affinity(selected_cpus)

        print(f"Restricted to CPUs: {selected_cpus}")
    else:
        print(
            "⚠️ Warning: CPU affinity setting is not supported on this platform. Proceeding without restriction."
        )


def clean_output_directory(output_dir, config_name, skip_model_creation=False):
    """
    Delete non-essential files in the output directory between configurations.
    Preserves:
    - KZG parameters file (kzg.params)
    - All CSV files (they contain benchmark results we need)
    - Model and input files if skip_model_creation is True
    """
    # Create the set of files to preserve
    preserve_files = {"kzg.params"}

    # Add all CSV files to the preserve list
    for file in output_dir.glob("*.csv"):
        preserve_files.add(file.name)

    # If skip_model_creation is True, preserve model and input files
    if skip_model_creation:
        preserve_files.add(MODEL)
        preserve_files.add(INPUT)

    # Loop through all files in the directory and delete those not in preserve_files
    for file in output_dir.iterdir():
        if file.is_file() and file.name not in preserve_files:
            print(f"Cleaning up: {file}")
            file.unlink()


def delete_all_csv_files(output_dir):
    """Delete all CSV files in the output directory at the start of benchmarking."""
    print("\n🗑️  Cleaning up previous benchmark results...")

    # Count how many files were deleted
    deleted_count = 0

    # Delete all CSV files in the output directory
    for csv_file in output_dir.glob("*.csv"):
        print(f"  Deleting: {csv_file}")
        csv_file.unlink()
        deleted_count += 1

    if deleted_count > 0:
        print(f"✅ Deleted {deleted_count} CSV files from previous runs.")
    else:
        print("✅ No previous benchmark CSV files found.")
    print("")


def calculate_average_accuracy(csv_file):
    """Calculate the average accuracy from a CSV file."""
    if not csv_file.exists():
        return None

    total_accuracy = 0
    row_count = 0

    with csv_file.open("r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_accuracy += int(row[ACCURACY])
            row_count += 1

    return total_accuracy / row_count if row_count > 0 else None


def compute_summary_statistics(output_dir, configs, run_ezkl, model_type):
    """Compute summary statistics for all configurations and return a DataFrame."""
    summary_data = []

    for num_dense, layer_width in configs:
        config_name = f"d{num_dense}_w{layer_width}" if model_type == "mlp" else "cnn"
        config_data = {
            "num_dense": num_dense,
            "layer_width": layer_width,
        }

        # Load ZKML data
        zkml_csv = output_dir / f"zkml_{config_name}.csv"
        if not zkml_csv.exists():
            print(f"❌ Error: ZKML CSV file {zkml_csv} does not exist.")
            sys.exit(1)

        try:
            zkml_df = pd.read_csv(zkml_csv)
            if not zkml_df.empty:
                config_data["zkml_accuracy"] = zkml_df[ACCURACY].mean()
                config_data["zkml_proving_time"] = zkml_df[PROVING].astype(
                    float).mean()
                config_data["zkml_verifying_time"] = zkml_df[VERIFYING].astype(
                    float).mean()
                if PROOF_SIZE in zkml_df.columns:
                    config_data["zkml_proof_size"] = zkml_df[
                        PROOF_SIZE].astype(float).mean()
        except Exception as e:
            print(f"❌ Error processing ZKML data for {config_name}: {e}")
            sys.exit(1)

        # Load PyTorch data
        pytorch_csv = output_dir / f"pytorch_{config_name}.csv"
        if pytorch_csv.exists():
            try:
                pytorch_df = pd.read_csv(pytorch_csv)
                if not pytorch_df.empty:
                    config_data["pytorch_accuracy"] = pytorch_df[
                        ACCURACY].mean()
            except Exception as e:
                print(
                    f"❌ Error processing PyTorch data for {config_name}: {e}")

        # Load EZKL data if requested
        if run_ezkl:
            ezkl_csv = output_dir / f"ezkl_{config_name}.csv"
            if not ezkl_csv.exists():
                print(f"❌ Error: EZKL CSV file {ezkl_csv} does not exist.")
                sys.exit(1)

            try:
                ezkl_df = pd.read_csv(ezkl_csv)
                if not ezkl_df.empty:
                    config_data["ezkl_accuracy"] = ezkl_df[ACCURACY].mean()
                    config_data["ezkl_proving_time"] = ezkl_df[PROVING].astype(
                        float).mean()
                    print(
                        f"EXTRACTED EZKL proving time: {ezkl_df[PROVING].astype(float)}"
                    )

                    # Add the new full proving time metric
                    if EZKL_FULL_PROVING in ezkl_df.columns:
                        config_data["ezkl_full_proving_time"] = ezkl_df[
                            EZKL_FULL_PROVING].astype(float).mean()
                        print(
                            f"FULL EZKL full proving time: {ezkl_df[EZKL_FULL_PROVING].astype(float)}"
                        )

                    config_data["ezkl_verifying_time"] = ezkl_df[
                        VERIFYING].astype(float).mean()
                    if PROOF_SIZE in ezkl_df.columns:
                        config_data["ezkl_proof_size"] = ezkl_df[
                            PROOF_SIZE].astype(float).mean()
            except Exception as e:
                print(f"❌ Error processing EZKL data for {config_name}: {e}")
                sys.exit(1)

        summary_data.append(config_data)

    return pd.DataFrame(summary_data)


def save_summary_csv(summary_df, output_dir):
    """Save summary statistics to a CSV file."""
    summary_path = output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to {summary_path}")
    return summary_path


def print_summary_table(summary_df, run_ezkl, show_accuracy=False):
    """Print summary statistics as a formatted table."""
    # Create a copy of the DataFrame for display formatting
    display_df = summary_df.copy()

    # Format times in milliseconds
    for col in display_df.columns:
        if "time" in col:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f} ms" if pd.notnull(x) else "N/A")

    # Format accuracy as percentage (only if we're showing accuracy)
    if show_accuracy and "zkml_accuracy" in display_df.columns:
        display_df["zkml_accuracy"] = display_df["zkml_accuracy"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    if show_accuracy and "ezkl_accuracy" in display_df.columns:
        display_df["ezkl_accuracy"] = display_df["ezkl_accuracy"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    if show_accuracy and "pytorch_accuracy" in display_df.columns:
        display_df["pytorch_accuracy"] = display_df["pytorch_accuracy"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")

    # Format proof sizes in KB
    if "zkml_proof_size" in display_df.columns:
        display_df["zkml_proof_size"] = display_df["zkml_proof_size"].apply(
            lambda x: f"{x:.2f} KB" if pd.notnull(x) else "N/A")
    if "ezkl_proof_size" in display_df.columns:
        display_df["ezkl_proof_size"] = display_df["ezkl_proof_size"].apply(
            lambda x: f"{x:.2f} KB" if pd.notnull(x) else "N/A")

    # Rename columns for better readability
    column_renames = {
        "num_dense": "Dense Layers",
        "layer_width": "Width",
        "zkml_accuracy": "ZKML Accuracy",
        "zkml_proving_time": "ZKML Proving",
        "zkml_verifying_time": "ZKML Verifying",
        "zkml_proof_size": "ZKML Proof Size",
        "ezkl_accuracy": "EZKL Accuracy",
        "ezkl_proving_time": "EZKL Proving",
        "ezkl_verifying_time": "EZKL Verifying",
        "ezkl_proof_size": "EZKL Proof Size",
        "pytorch_accuracy": "PyTorch Accuracy",
        "ezkl_full_proving_time": "EZKL Full Proving",
    }
    display_df = display_df.rename(columns=column_renames)

    # Select columns based on what's available, grouped by metric type
    # First the configuration columns
    columns_to_show = ["Dense Layers", "Width"]

    # Group all accuracy columns together (only if we're showing accuracy)
    if show_accuracy:
        accuracy_columns = ["PyTorch Accuracy", "ZKML Accuracy"]
        if run_ezkl:
            accuracy_columns.append("EZKL Accuracy")
        columns_to_show.extend(accuracy_columns)

    # Group all proving time columns together
    proving_columns = ["ZKML Proving"]
    if run_ezkl:
        proving_columns.extend(["EZKL Proving", "EZKL Full Proving"])
    columns_to_show.extend(proving_columns)

    # Group all verification time columns together
    verify_columns = ["ZKML Verifying"]
    if run_ezkl:
        verify_columns.append("EZKL Verifying")
    columns_to_show.extend(verify_columns)

    # Group all proof size columns together
    size_columns = ["ZKML Proof Size"]
    if run_ezkl:
        size_columns.append("EZKL Proof Size")
    columns_to_show.extend(size_columns)

    # Only include columns that actually exist in the dataframe
    columns_to_show = [
        col for col in columns_to_show if col in display_df.columns
    ]

    # Print the table
    print(
        tabulate.tabulate(display_df[columns_to_show],
                          headers="keys",
                          tablefmt="grid"))
    print("\n")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for ZKML and EZKL")
    parser.add_argument(
        "--configs",
        type=str,
        default="1,10",
        help="Model configurations for mlp to benchmark in format 'dense,width:dense,width'"
    )
    parser.add_argument("--repeats",
                        type=int,
                        default=3,
                        help="Number of times to repeat each benchmark")
    parser.add_argument("--run-ezkl",
                        action="store_true",
                        help="Run EZKL benchmarks (slower)")
    parser.add_argument(
        "--skip-ezkl-calibration",
        action="store_true",
        help="Skip the EZKL calibration step (faster but potentially less accurate)"
    )
    parser.add_argument(
        "--ezkl-check-mode",
        choices=["safe", "unsafe"],
        default="safe",
        help="Set EZKL check mode (safe or unsafe) - safe by default")
    parser.add_argument(
        "--show-accuracy",
        action="store_true",
        help="Show accuracy information in benchmark output (off by default)")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("bench"),
                        help="Directory to save benchmark results")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Limit the number of threads used (default: no limit)")
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of input/output samples to process for both ZKML and EZKL (default: 30)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mlp", "cnn", "covid"],
        default="mlp",
        help="Type of model to benchmark: 'mlp' for MLP, 'cnn' for CNN (default: mlp)"
    )
    parser.add_argument(
        "--skip-model-creation",
        action="store_true",
        help="Skip model creation and use existing model files and input data (default: False, will regenerate models)"
    )
    parser.add_argument(
        "--param-cnn",
        type=int,
        default=None,
        help="Target parameter count for CNN model (only used when model-type is cnn) 62k default"
    )
    parser.add_argument(
        "--skip-proving",
        action="store_true",
        help="Skip proving and verifying steps in ZKML (faster but no ZK proof)"
    )
    return parser.parse_args()


def setup_environment(args):
    """Setup the environment based on the provided arguments."""
    if args.num_threads is not None:
        set_cpu_affinity(args.num_threads)


def run_configurations(configs, args):
    """Run all specified configurations with the specified number of repeats."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    zkml_csv_files = []
    ezkl_csv_files = []
    pytorch_csv_files = []

    for config in configs:
        num_dense, layer_width = config
        config_name = f"d{num_dense}_w{layer_width}" if args.model_type == "mlp" else "cnn"

        # Clean up the output directory for this configuration
        clean_output_directory(output_dir, config_name,
                               args.skip_model_creation)

        for run_idx in range(args.repeats):
            # Pass all EZKL-related parameters and CNN parameter count
            zkml_csv, ezkl_csv, pytorch_csv = run_benchmark(
                num_dense, layer_width, run_idx, output_dir, args.verbose,
                args.run_ezkl, args.samples, args.model_type,
                args.skip_ezkl_calibration, args.ezkl_check_mode,
                args.show_accuracy, args.skip_model_creation, args.param_cnn,
                args.skip_proving)

            if zkml_csv:
                zkml_csv_files.append(zkml_csv)
            if ezkl_csv:
                ezkl_csv_files.append(ezkl_csv)
            if pytorch_csv:
                pytorch_csv_files.append(pytorch_csv)

    return zkml_csv_files, ezkl_csv_files, pytorch_csv_files


def calculate_and_print_results(configs, zkml_csv_files, ezkl_csv_files,
                                pytorch_csv_files, args):
    """Calculate and print average accuracy for all runs."""
    # If show-accuracy is not enabled, just return without printing
    if not args.show_accuracy:
        return

    for config_name in configs:
        zkml_csv = Path(
            args.output_dir) / f"zkml_d{config_name[0]}_w{config_name[1]}.csv"
        ezkl_csv = Path(
            args.output_dir) / f"ezkl_d{config_name[0]}_w{config_name[1]}.csv"
        pytorch_csv = Path(
            args.output_dir
        ) / f"pytorch_d{config_name[0]}_w{config_name[1]}.csv"

        zkml_accuracy = calculate_average_accuracy(zkml_csv)
        ezkl_accuracy = calculate_average_accuracy(
            ezkl_csv) if args.run_ezkl else None
        pytorch_accuracy = calculate_average_accuracy(
            pytorch_csv) if pytorch_csv.exists() else None

        if zkml_accuracy is not None:
            print(
                f"Average accuracy for ZKML (d{config_name[0]}_w{config_name[1]}): {zkml_accuracy:.2f}"
            )

        if ezkl_accuracy is not None:
            print(
                f"Average accuracy for EZKL (d{config_name[0]}_w{config_name[1]}): {ezkl_accuracy:.2f}"
            )

        if pytorch_accuracy is not None:
            print(
                f"Average accuracy for PyTorch (d{config_name[0]}_w{config_name[1]}): {pytorch_accuracy:.2f}"
            )


def main():
    """Main function to run benchmarks."""
    args = parse_arguments()
    setup_environment(args)

    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete all CSV files at the beginning
    delete_all_csv_files(output_dir)

    # Parse model configurations
    configs = []
    for config in args.configs.split(':'):
        parts = config.split(',')
        if len(parts) != 2:
            print(f"❌ Error: Invalid configuration format: {config}")
            sys.exit(1)

        num_dense = int(parts[0])
        layer_width = int(parts[1])
        configs.append((num_dense, layer_width))

    # Run all specified configurations
    zkml_csv_files, ezkl_csv_files, pytorch_csv_files = run_configurations(
        configs, args)

    # Calculate and print results
    calculate_and_print_results(configs, zkml_csv_files, ezkl_csv_files,
                                pytorch_csv_files, args)

    # Generate and print summary statistics
    summary_df = compute_summary_statistics(args.output_dir, configs,
                                            args.run_ezkl, args.model_type)
    _ = save_summary_csv(summary_df, args.output_dir)
    print_summary_table(summary_df, args.run_ezkl, args.show_accuracy)


if __name__ == "__main__":
    main()
