#!/usr/bin/env python3
"""
Run frequent subgraph mining algorithms (gSpan, FSG, Gaston) and generate runtime plots.
"""

import os
import sys
import subprocess
import time
import re
import matplotlib.pyplot as plt


def run_algorithm(executable, args, output_file, timeout=3600):
    """
    Run an algorithm and measure its runtime.
    Returns (runtime_seconds, success, stdout, stderr)
    """
    cmd = [executable] + args
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        end_time = time.time()
        runtime = end_time - start_time
        
        # Save output if specified
        if output_file and result.stdout:
            with open(output_file, 'w') as f:
                f.write(result.stdout)
        
        if result.returncode != 0:
            print(f"Warning: Command returned non-zero exit code: {result.returncode}")
            if result.stderr:
                print(f"Stderr: {result.stderr[:500]}")
        
        return runtime, True, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Timeout after {timeout} seconds")
        return timeout, False, "", ""
    except FileNotFoundError:
        print(f"Executable not found: {executable}")
        return 0, False, "", ""
    except Exception as e:
        print(f"Error running command: {e}")
        return 0, False, "", ""


def run_gspan(executable, dataset_path, support_percent, output_path):
    """
    Run gSpan with given support threshold.
    gSpan -f filename -s frequency [-o] [-i]
    frequency is a float (0.0 to 1.0)
    Output is saved to filename.fp
    """
    support_frac = support_percent / 100.0
    
    # gSpan saves output to <input>.fp, so we'll copy it after
    args = ['-f', dataset_path, '-s', str(support_frac), '-o']
    
    runtime, success, stdout, stderr = run_algorithm(executable, args, None)
    
    # Copy the output file
    gspan_output = dataset_path + '.fp'
    if os.path.exists(gspan_output):
        with open(gspan_output, 'r') as src:
            content = src.read()
        with open(output_path, 'w') as dst:
            dst.write(content)
        os.remove(gspan_output)  # Clean up
    else:
        # Create empty file if no output
        open(output_path, 'w').close()
    
    return runtime, success


def run_fsg(executable, dataset_path, support_percent, output_path):
    """
    Run FSG with given support threshold.
    fsg -s support dataset
    support is in percentage (integer)
    FSG outputs to yeast_fsg.fp in the current directory
    """
    args = ['-s', str(int(support_percent)), dataset_path]
    
    runtime, success, stdout, stderr = run_algorithm(executable, args, None)
    
    # FSG outputs to yeast_fsg.fp (basename of dataset with .fp extension)
    # Get the base name without extension for the FSG output file
    dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]
    fsg_output = dataset_basename + '.fp'
    
    # Check both in current directory and dataset directory
    possible_locations = [
        fsg_output,
        os.path.join(os.path.dirname(dataset_path), fsg_output),
        dataset_path + '.fp'
    ]
    
    found = False
    for fsg_output_path in possible_locations:
        if os.path.exists(fsg_output_path):
            with open(fsg_output_path, 'r') as src:
                content = src.read()
            with open(output_path, 'w') as dst:
                dst.write(content)
            os.remove(fsg_output_path)
            found = True
            break
    
    if not found:
        # Create empty file if no output found
        open(output_path, 'w').close()
    
    return runtime, success


def run_gaston(executable, dataset_path, support_percent, output_path, num_graphs):
    """
    Run Gaston with given support threshold.
    gaston support input_file output_file
    support is absolute count (not percentage)
    """
    # Convert percentage to absolute count
    support_count = max(1, int(num_graphs * support_percent / 100.0))
    
    args = [str(support_count), dataset_path, output_path]
    
    runtime, success, stdout, stderr = run_algorithm(executable, args, None)
    
    # If output file doesn't exist, create empty
    if not os.path.exists(output_path):
        open(output_path, 'w').close()
    
    return runtime, success


def count_graphs(dataset_path):
    """Count the number of graphs in a gSpan/Gaston format dataset."""
    count = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.startswith('t #'):
                count += 1
    return count


def generate_plot(results, output_path):
    """
    Generate runtime comparison plot.
    results: dict of {algorithm: {support: runtime}}
    """
    plt.figure(figsize=(10, 6))
    
    supports = [5, 10, 25, 50, 95]
    
    markers = {'gspan': 'o-', 'fsg': 's-', 'gaston': '^-'}
    colors = {'gspan': 'blue', 'fsg': 'green', 'gaston': 'red'}
    labels = {'gspan': 'gSpan', 'fsg': 'FSG', 'gaston': 'Gaston'}
    
    for algo in ['gspan', 'fsg', 'gaston']:
        if algo in results:
            runtimes = [results[algo].get(s, 0) for s in supports]
            plt.plot(supports, runtimes, markers[algo], color=colors[algo], 
                    label=labels[algo], linewidth=2, markersize=8)
    
    plt.xlabel('Minimum Support (%)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Frequent Subgraph Mining: Runtime Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(supports)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    if len(sys.argv) != 6:
        print("Usage: python run_experiments.py <gspan_exec> <fsg_exec> <gaston_exec> <dataset> <output_dir>")
        sys.exit(1)
    
    gspan_exec = sys.argv[1]
    fsg_exec = sys.argv[2]
    gaston_exec = sys.argv[3]
    dataset_path = sys.argv[4]
    output_dir = sys.argv[5]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Support thresholds
    supports = [5, 10, 25, 50, 95]
    
    # Prepare converted datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converted_dir = os.path.join(script_dir, 'converted_data')
    os.makedirs(converted_dir, exist_ok=True)
    
    # Convert dataset
    print("Converting dataset to different formats...")
    convert_cmd = [
        sys.executable,
        os.path.join(script_dir, 'convert_dataset.py'),
        dataset_path,
        converted_dir
    ]
    subprocess.run(convert_cmd, check=True)
    
    gspan_data = os.path.join(converted_dir, 'yeast_gspan.txt')
    gaston_data = os.path.join(converted_dir, 'yeast_gaston.txt')
    fsg_data = os.path.join(converted_dir, 'yeast_fsg.txt')
    
    # Count graphs for Gaston
    num_graphs = count_graphs(gspan_data)
    print(f"Number of graphs in dataset: {num_graphs}")
    
    results = {'gspan': {}, 'fsg': {}, 'gaston': {}}
    
    # Run experiments
    for support in supports:
        print(f"\n{'='*50}")
        print(f"Running with support = {support}%")
        print('='*50)
        
        # Run gSpan
        print("\n--- gSpan ---")
        output_file = os.path.join(output_dir, f'gspan{support}')
        runtime, success = run_gspan(gspan_exec, gspan_data, support, output_file)
        results['gspan'][support] = runtime
        print(f"gSpan runtime: {runtime:.2f}s")
        
        # Run FSG
        print("\n--- FSG ---")
        output_file = os.path.join(output_dir, f'fsg{support}')
        runtime, success = run_fsg(fsg_exec, fsg_data, support, output_file)
        results['fsg'][support] = runtime
        print(f"FSG runtime: {runtime:.2f}s")
        
        # Run Gaston
        print("\n--- Gaston ---")
        output_file = os.path.join(output_dir, f'gaston{support}')
        runtime, success = run_gaston(gaston_exec, gaston_data, support, output_file, num_graphs)
        results['gaston'][support] = runtime
        print(f"Gaston runtime: {runtime:.2f}s")
    
    # Generate plot
    print("\n" + "="*50)
    print("Generating plot...")
    plot_path = os.path.join(output_dir, 'plot.png')
    generate_plot(results, plot_path)
    
    # Print summary
    print("\n" + "="*50)
    print("Summary of Results:")
    print("="*50)
    print(f"{'Support':<10} {'gSpan (s)':<15} {'FSG (s)':<15} {'Gaston (s)':<15}")
    print("-"*55)
    for support in supports:
        print(f"{support}%{'':<8} {results['gspan'][support]:<15.2f} {results['fsg'][support]:<15.2f} {results['gaston'][support]:<15.2f}")


if __name__ == "__main__":
    main()
