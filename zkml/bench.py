import subprocess
import os
import time
import sys
from pathlib import Path
import argparse
import shutil
import sys
import csv
import logging
import subprocess
from typing import Callable, Dict, List, Union
import re
import json
import itertools
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
                raise RuntimeError(f"Command {task} failed with exit code {result.returncode}: {result}")
        else:
            result = task()
        
        elapsed = int((time.perf_counter() - start_time) * 1000)  # Convert to ms
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
        print(f"❌ Error: '{command}' is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)  # Exit with an error code

def ex(cmd, verbose=False):
    print(f"\n🔄 Running command: {' '.join(cmd)}")
    start_time = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.perf_counter()
    
    elapsed_time_ms = (end_time - start_time) * 1000
    
    if verbose and result.stdout:
        print(f"📝 Output:\n{result.stdout}")
    
    if result.returncode != 0:
        print(f"❌ Error (code {result.returncode}):\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    
    print(f"⏱️  Took {elapsed_time_ms:.0f}ms")
    
    return {
        "stdout": result.stdout,
        "elapsed_time_ms": elapsed_time_ms
    }

# Default paths
DEFAULT_BENCH_FOLDER = Path("./bench/")
PYTORCH_SCRIPT = "./assets/scripts/MLP/mlp.py"
MODEL = "mlp-model.onnx"
INPUT = "mlp-input.json"
EZKL_KZG_PARAMS = "kzg.params"

def create_model(num_dense, layer_width, output_dir, verbose, num_samples):
    """Create a PyTorch model with the specified parameters"""
    print(f"Creating PyTorch model: dense={num_dense}, width={layer_width}")
    ex(["python3", PYTORCH_SCRIPT,
        "--num-dense", str(num_dense),
        "--layer-width", str(layer_width),
        "--export", str(output_dir),
        "--num-samples", str(num_samples)],
       verbose=verbose)

def run_zkml_benchmark(config_name, output_dir, verbose, num_samples):
    """Run ZKML benchmark and save results to CSV"""
    zkml_csv = output_dir / f"zkml_{config_name}.csv"
    
    print(f"Running ZKML benchmark for {config_name}")
    ensure_command_exists("cargo")
    out = ex(["cargo", "run", "--release", "--", 
              "-i", str(output_dir / INPUT),
              "-o", str(output_dir / MODEL),
              "--bench", str(zkml_csv),
              "--num-samples", str(num_samples)], verbose=verbose)
    print("ZKML benchmark completed")
    return zkml_csv

def run_ezkl_benchmark(config_name, run_index, output_dir, verbose, num_samples):
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
        print(f"Found {available_samples} input/output pairs in {INPUT}, will process {max_samples}")
        
        # Run setup steps once (these don't depend on the input data)
        ex(["ezkl", "gen-settings", "-K", str(LOGROWS),"-M", MODEL], verbose=verbose)
        
        # Use the timing from ex() function directly
        calibration_result = ex(["ezkl", "calibrate-settings", "-M", MODEL, "-D", INPUT,"--max-logrows", str(LOGROWS)], verbose=verbose)
        calibration_time = calibration_result["elapsed_time_ms"]
        
        if not Path(EZKL_KZG_PARAMS).exists():
            print("Downloading SRS params")
            ex(["ezkl", "get-srs", "--logrows", str(LOGROWS),"--srs-path", EZKL_KZG_PARAMS], verbose=verbose)
        ex(["ezkl", "compile-circuit", "-M", MODEL], verbose=verbose)
        
        # Run setup once and measure time
        setup_start = time.perf_counter()
        ex(["ezkl", "setup", "--srs-path", EZKL_KZG_PARAMS], verbose=verbose)
        setup_time_ms = (time.perf_counter() - setup_start) * 1000
        
        # Process each input/output pair up to the limit
        for sample_idx in range(max_samples):
            print(f"\n[+] Processing EZKL sample {sample_idx+1}/{max_samples}")
            
            # Create a temporary JSON file with just this input/output pair
            temp_input = {
                "input_data": [original_input_data["input_data"][sample_idx]],
                "output_data": [original_input_data["output_data"][sample_idx]]
            }
            
            temp_input_file = f"temp_input_{sample_idx}.json"
            with open(temp_input_file, "w") as f:
                json.dump(temp_input, f)
            
            # Initialize bencher for this sample
            bencher = CSVBencher([CONFIG, RUN, SAMPLE, SETUP, INFERENCE, PROVING, VERIFYING, ACCURACY, PROOF_SIZE])
            bencher.set(CONFIG, config_name)
            bencher.set(RUN, str(run_index))
            bencher.set(SAMPLE, str(sample_idx))
            bencher.set(SETUP, str(setup_time_ms))
            
            # Capture the time for gen-witness
            witness_result = ex(["ezkl", "gen-witness", "-D", temp_input_file], verbose=verbose)
            witness_time_ms = witness_result["elapsed_time_ms"]
            bencher.set(INFERENCE, str(witness_time_ms))
            
            # For proving, extract the specific timing
            proving_result = ex(["ezkl", "prove", "--srs-path", EZKL_KZG_PARAMS], verbose=verbose)
            
            # Extract the proof time using regex
            proof_time_match = re.search(r"\[.*ezkl::pfsys\] - proof took (\d+\.\d+)", proving_result["stdout"])
            if proof_time_match:
                proof_time_seconds = float(proof_time_match.group(1))
                proof_time_ms = int(proof_time_seconds * 1000)
                print(f"Extracted proof time: {proof_time_ms}ms")
            else:
                print("Could not extract proof time, using full command time")
                proof_time_ms = proving_result["elapsed_time_ms"]
            
            # Sum the witness and proof times
            total_proving_time_ms = witness_time_ms + proof_time_ms
            bencher.set(PROVING, f"{total_proving_time_ms:.2f}")
            
            # Run verification
            bencher.r(VERIFYING, ["ezkl", "verify", "--srs-path", EZKL_KZG_PARAMS])
            
            # Extract outputs and check accuracy
            with open("proof.json", "r") as f:
                proof_data = json.load(f)
            
            ezkl_outputs = [float(x) for x in proof_data["pretty_public_inputs"]["rescaled_outputs"][0]]
            ezkl_argmax = ezkl_outputs.index(max(ezkl_outputs))
            print(f"EZKL outputs len: {len(proof_data['pretty_public_inputs']['rescaled_outputs'])}")
            print(f"EZKL output: {ezkl_outputs}, argmax: {ezkl_argmax}")
            
            pytorch_outputs = [float(x) for x in temp_input["output_data"][0]]
            pytorch_argmax = pytorch_outputs.index(max(pytorch_outputs))
            print(f"PyTorch output: {pytorch_outputs} (len {len(pytorch_outputs)}), argmax: {pytorch_argmax}")
            
            is_correct = 1 if ezkl_argmax == pytorch_argmax else 0
            bencher.set(ACCURACY, str(is_correct))
            print(f"Correctness check: {'PASS' if is_correct else 'FAIL'}")
            
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

def run_benchmark(num_dense, layer_width, run_index, output_dir, verbose, run_ezkl, num_samples):
    """Run a single benchmark with the specified parameters"""
    config_name = f"d{num_dense}_w{layer_width}"
    
    print(f"\n{'='*80}")
    print(f"Running benchmark: dense={num_dense}, width={layer_width}, run={run_index}")
    print(f"{'='*80}\n")
    
    # Step 1: Create PyTorch model with specified number of samples
    create_model(num_dense, layer_width, output_dir, verbose, num_samples)
    
    # Step 2: Run ZKML benchmark
    zkml_csv = run_zkml_benchmark(config_name, output_dir, verbose, num_samples)
    
    # Step 3: Calculate PyTorch accuracy from the input_output.json
    pytorch_csv = calculate_pytorch_accuracy(config_name, run_index, output_dir)
    
    # Step 4: Run EZKL benchmark if requested
    ezkl_csv = None
    if run_ezkl:
        ezkl_csv = run_ezkl_benchmark(config_name, run_index, output_dir, verbose, num_samples)
        print(f"Results saved to {zkml_csv}, {pytorch_csv}, and {ezkl_csv}")
    else:
        print(f"Results saved to {zkml_csv} and {pytorch_csv} (EZKL comparison skipped)")
        
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
        print(f"❌ Error: input/output JSON does not contain PyTorch outputs")
        print(f"Please regenerate the input file with an updated script that includes pytorch_output.")
        sys.exit(1)
    
    # Calculate accuracy for each sample
    for i, (expected_output, pytorch_output) in enumerate(zip(data["output_data"], data["pytorch_output"])):
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
        print(f"❌ Error: Requested {max_threads} threads, but only {num_logical_processors} logical processors are available.")
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
        print("⚠️ Warning: CPU affinity setting is not supported on this platform. Proceeding without restriction.")

def delete_csv_files(output_dir, config_name):
    """Delete existing CSV files for the given configuration."""
    zkml_csv = output_dir / f"zkml_{config_name}.csv"
    ezkl_csv = output_dir / f"ezkl_{config_name}.csv"
    
    for csv_file in [zkml_csv, ezkl_csv]:
        if csv_file.exists():
            csv_file.unlink()
            print(f"Deleted existing file: {csv_file}")

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

def compute_summary_statistics(output_dir, configs, run_ezkl):
    """Compute summary statistics for all configurations and return a DataFrame."""
    summary_data = []
    
    for num_dense, layer_width in configs:
        config_name = f"d{num_dense}_w{layer_width}"
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
                config_data["zkml_proving_time"] = zkml_df[PROVING].astype(float).mean()
                config_data["zkml_verifying_time"] = zkml_df[VERIFYING].astype(float).mean()
                if PROOF_SIZE in zkml_df.columns:
                    config_data["zkml_proof_size"] = zkml_df[PROOF_SIZE].astype(float).mean()
        except Exception as e:
            print(f"❌ Error processing ZKML data for {config_name}: {e}")
            sys.exit(1)
        
        # Load PyTorch data
        pytorch_csv = output_dir / f"pytorch_{config_name}.csv"
        if pytorch_csv.exists():
            try:
                pytorch_df = pd.read_csv(pytorch_csv)
                if not pytorch_df.empty:
                    config_data["pytorch_accuracy"] = pytorch_df[ACCURACY].mean()
            except Exception as e:
                print(f"❌ Error processing PyTorch data for {config_name}: {e}")
        
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
                    config_data["ezkl_proving_time"] = ezkl_df[PROVING].astype(float).mean()
                    config_data["ezkl_verifying_time"] = ezkl_df[VERIFYING].astype(float).mean()
                    if PROOF_SIZE in ezkl_df.columns:
                        config_data["ezkl_proof_size"] = ezkl_df[PROOF_SIZE].astype(float).mean()
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

def print_summary_table(summary_df, run_ezkl):
    """Print summary statistics as a formatted table."""
    # Create a copy of the DataFrame for display formatting
    display_df = summary_df.copy()
    
    # Format times in milliseconds
    for col in display_df.columns:
        if "time" in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f} ms" if pd.notnull(x) else "N/A")
    
    # Format accuracy as percentage
    if "zkml_accuracy" in display_df.columns:
        display_df["zkml_accuracy"] = display_df["zkml_accuracy"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    if "ezkl_accuracy" in display_df.columns:
        display_df["ezkl_accuracy"] = display_df["ezkl_accuracy"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    if "pytorch_accuracy" in display_df.columns:
        display_df["pytorch_accuracy"] = display_df["pytorch_accuracy"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    
    # Format proof sizes in KB
    if "zkml_proof_size" in display_df.columns:
        display_df["zkml_proof_size"] = display_df["zkml_proof_size"].apply(lambda x: f"{x:.2f} KB" if pd.notnull(x) else "N/A")
    if "ezkl_proof_size" in display_df.columns:
        display_df["ezkl_proof_size"] = display_df["ezkl_proof_size"].apply(lambda x: f"{x:.2f} KB" if pd.notnull(x) else "N/A")
    
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
        "pytorch_accuracy": "PyTorch Accuracy"
    }
    display_df = display_df.rename(columns=column_renames)
    
    # Select columns based on what's available, grouped by metric type
    # First the configuration columns
    columns_to_show = ["Dense Layers", "Width"]
    
    # Group all accuracy columns together
    accuracy_columns = ["PyTorch Accuracy", "ZKML Accuracy"]
    if run_ezkl:
        accuracy_columns.append("EZKL Accuracy")
    columns_to_show.extend(accuracy_columns)
    
    # Group all proving time columns together
    proving_columns = ["ZKML Proving"]
    if run_ezkl:
        proving_columns.append("EZKL Proving")
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
    columns_to_show = [col for col in columns_to_show if col in display_df.columns]
    
    # Print the table
    print(tabulate.tabulate(display_df[columns_to_show], headers="keys", tablefmt="grid"))
    print("\n")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmarking Tool for ZKML and EZKL")
    parser.add_argument("--configs", type=str, default="3,4:4,8", 
                        help="Configurations to run as 'dense1,width1:dense2,width2:...'")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of times to repeat each configuration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCH_FOLDER,
                        help="Directory to store benchmark results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for each command")
    parser.add_argument("--run-ezkl", action="store_true",
                        help="Enable EZKL comparison (off by default)")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="Limit the number of threads used (default: no limit)")
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of input/output samples to process for both ZKML and EZKL (default: 30)")
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
        config_name = f"d{num_dense}_w{layer_width}"
        
        # Delete existing CSV files for this configuration
        delete_csv_files(output_dir, config_name)
        
        for run_idx in range(args.repeats):
            # Pass args.samples to run_benchmark
            zkml_csv, ezkl_csv, pytorch_csv = run_benchmark(
                num_dense, layer_width, run_idx, output_dir, 
                args.verbose, args.run_ezkl, args.samples
            )
            
            if zkml_csv:
                zkml_csv_files.append(zkml_csv)
            if ezkl_csv:
                ezkl_csv_files.append(ezkl_csv)
            if pytorch_csv:
                pytorch_csv_files.append(pytorch_csv)
    
    return zkml_csv_files, ezkl_csv_files, pytorch_csv_files

def calculate_and_print_results(configs, zkml_csv_files, ezkl_csv_files, pytorch_csv_files, args):
    """Calculate and print average accuracy for all runs."""
    for config_name in configs:
        zkml_csv = Path(args.output_dir) / f"zkml_d{config_name[0]}_w{config_name[1]}.csv"
        ezkl_csv = Path(args.output_dir) / f"ezkl_d{config_name[0]}_w{config_name[1]}.csv"
        pytorch_csv = Path(args.output_dir) / f"pytorch_d{config_name[0]}_w{config_name[1]}.csv"
        
        zkml_accuracy = calculate_average_accuracy(zkml_csv)
        ezkl_accuracy = calculate_average_accuracy(ezkl_csv) if args.run_ezkl else None
        pytorch_accuracy = calculate_average_accuracy(pytorch_csv) if pytorch_csv.exists() else None
        
        if zkml_accuracy is not None:
            print(f"Average accuracy for ZKML (d{config_name[0]}_w{config_name[1]}): {zkml_accuracy:.2f}")
        
        if ezkl_accuracy is not None:
            print(f"Average accuracy for EZKL (d{config_name[0]}_w{config_name[1]}): {ezkl_accuracy:.2f}")
        
        if pytorch_accuracy is not None:
            print(f"Average accuracy for PyTorch (d{config_name[0]}_w{config_name[1]}): {pytorch_accuracy:.2f}")

def main():
    args = parse_arguments()
    setup_environment(args)
    
    # Parse configurations
    configs = []
    for config_str in args.configs.split(':'):
        parts = config_str.split(',')
        if len(parts) != 2:
            print(f"Invalid configuration: {config_str}. Expected format: 'dense,width'")
            sys.exit(1)
        try:
            num_dense = int(parts[0])
            layer_width = int(parts[1])
            configs.append((num_dense, layer_width))
        except ValueError:
            print(f"Invalid configuration values in: {config_str}. Expected integers.")
            sys.exit(1)
    
    print(f"Running {len(configs)} configurations, each repeated {args.repeats} times")
    
    zkml_csv_files, ezkl_csv_files, pytorch_csv_files = run_configurations(configs, args)
    calculate_and_print_results(configs, zkml_csv_files, ezkl_csv_files, pytorch_csv_files, args)
    
    # Generate and print summary statistics
    summary_df = compute_summary_statistics(args.output_dir, configs, args.run_ezkl)
    summary_path = save_summary_csv(summary_df, args.output_dir)
    print_summary_table(summary_df, args.run_ezkl)

if __name__ == "__main__":
    main()