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

logging.basicConfig(level=logging.INFO)

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

def create_model(num_dense, layer_width, output_dir, verbose):
    """Create a PyTorch model with the specified parameters"""
    print(f"Creating PyTorch model: dense={num_dense}, width={layer_width}")
    ex(["python3", PYTORCH_SCRIPT,
        "--num-dense", str(num_dense),
        "--layer-width", str(layer_width),
        "--export", str(output_dir)], verbose=verbose)

def run_zkml_benchmark(config_name, output_dir, verbose):
    """Run ZKML benchmark and save results to CSV"""
    zkml_csv = output_dir / f"zkml_{config_name}.csv"
    
    print(f"Running ZKML benchmark for {config_name}")
    ensure_command_exists("cargo")
    out = ex(["cargo", "run", "--release", "--", 
              "-i", str(output_dir / INPUT),
              "-o", str(output_dir / MODEL),
              "--bench", str(zkml_csv)], verbose=verbose)
    print("ZKML benchmark completed")
    return zkml_csv

def run_ezkl_benchmark(config_name, run_index, output_dir, verbose):
    """Run EZKL benchmark and save results to CSV"""
    # Create absolute paths before changing directory
    ezkl_csv = output_dir / f"ezkl_{config_name}.csv"
    absolute_ezkl_csv = ezkl_csv.absolute()
    
    print(f"Running EZKL benchmark for {config_name}, run {run_index}")
    print(f"Moving to {output_dir} for ezkl bench")
    original_dir = os.getcwd()
    os.chdir(output_dir)
    ensure_command_exists("ezkl")
    
    try:
        SETUP = "setup (ms)" 
        INFERENCE = "inference (ms)"
        PROVING = "proving (ms)"
        VERIFYING = "verifying (ms)"
        ACCURACY = "accuracy (bool)"
        PROOF_SIZE = "proof size (KB)"
        CONFIG = "config"
        RUN = "run"
        LOGROWS = 24
        
        bencher = CSVBencher([CONFIG, RUN, SETUP, INFERENCE, PROVING, VERIFYING, ACCURACY, PROOF_SIZE])
        bencher.set(CONFIG, config_name)
        bencher.set(RUN, str(run_index))
        
        # Run setup steps
        ex(["ezkl", "gen-settings", "-K", str(LOGROWS),"-M", MODEL], verbose=verbose)
        ex(["ezkl", "calibrate-settings", "-M", MODEL, "-D", INPUT], verbose=verbose)
        if not Path(EZKL_KZG_PARAMS).exists():
            print("Downloading SRS params")
            ex(["ezkl", "get-srs", "--logrows", str(LOGROWS),"--srs-path", EZKL_KZG_PARAMS], verbose=verbose)
        ex(["ezkl", "compile-circuit", "-M", MODEL], verbose=verbose)
        
        # Run benchmarks
        bencher.r(SETUP, ["ezkl", "setup", "--srs-path", EZKL_KZG_PARAMS])
        
        # Capture the time for gen-witness
        witness_result = ex(["ezkl", "gen-witness", "-D", INPUT], verbose=verbose)
        witness_time_ms = witness_result["elapsed_time_ms"]
        
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
        bencher.set(PROVING, str(total_proving_time_ms))
        
        bencher.r(VERIFYING, ["ezkl", "verify", "--srs-path", EZKL_KZG_PARAMS])
        
        # Extract outputs and check accuracy
        with open("proof.json", "r") as f:
            proof_data = json.load(f)
        
        ezkl_outputs = [float(x) for x in proof_data["pretty_public_inputs"]["rescaled_outputs"][0]]
        ezkl_argmax = ezkl_outputs.index(max(ezkl_outputs))
        print(f"EZKL output: {ezkl_outputs}, argmax: {ezkl_argmax}")
        
        with open(INPUT, "r") as f:
            input_data = json.load(f)
        
        pytorch_outputs = [float(x) for x in input_data["output_data"][0]]
        pytorch_argmax = pytorch_outputs.index(max(pytorch_outputs))
        print(f"PyTorch output: {pytorch_outputs}, argmax: {pytorch_argmax}")
        
        is_correct = 1 if ezkl_argmax == pytorch_argmax else 0
        bencher.set(ACCURACY, str(is_correct))
        print(f"Correctness check: {'PASS' if is_correct else 'FAIL'}")
        
        # Extract proof size in KB
        proof_size_kb = len(proof_data["proof"]) / 1024.0
        bencher.set(PROOF_SIZE, f"{proof_size_kb:.3f}")
        print(f"Proof size: {proof_size_kb:.3f} KB")
        
        # Use absolute path for CSV file
        bencher.flush(absolute_ezkl_csv)
        
        return ezkl_csv
    finally:
        # Always return to the original directory
        os.chdir(original_dir)

def run_benchmark(num_dense, layer_width, run_index, output_dir, verbose, run_ezkl):
    """Run a single benchmark with the specified parameters"""
    config_name = f"d{num_dense}_w{layer_width}"
    
    print(f"\n{'='*80}")
    print(f"Running benchmark: dense={num_dense}, width={layer_width}, run={run_index}")
    print(f"{'='*80}\n")
    
    # Step 1: Create PyTorch model
    create_model(num_dense, layer_width, output_dir, verbose)
    
    # Step 2: Run ZKML benchmark
    zkml_csv = run_zkml_benchmark(config_name, output_dir, verbose)
    
    # Step 3: Conditionally Run EZKL benchmark
    if run_ezkl:
        ezkl_csv = run_ezkl_benchmark(config_name, run_index, output_dir, verbose)
        print(f"Results saved to {zkml_csv} and {ezkl_csv}")
    else:
        print(f"Results saved to {zkml_csv} (EZKL comparison skipped)")


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

def main():
    parser = argparse.ArgumentParser(description="Run multiple MLP benchmarks")
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
    
    args = parser.parse_args()
    
    # Set CPU affinity if num-threads is specified
    if args.num_threads is not None:
        set_cpu_affinity(args.num_threads)
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all configurations
    for config_idx, (num_dense, layer_width) in enumerate(configs):
        for run_idx in range(args.repeats):
            print(f"\nRunning configuration {config_idx+1}/{len(configs)}, "
                  f"repeat {run_idx+1}/{args.repeats}")
            run_benchmark(num_dense, layer_width, run_idx, output_dir, args.verbose, args.run_ezkl)

if __name__ == "__main__":
    main()