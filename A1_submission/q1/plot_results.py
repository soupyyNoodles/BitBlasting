#!/usr/bin/env python3
"""
Script to plot runtime comparison of Apriori and FP-Growth algorithms
"""

import sys
import matplotlib.pyplot as plt

def plot_results(results_file, output_file):
    """
    Plot runtime vs support threshold for Apriori and FP-Growth
    
    Args:
        results_file: Path to results.txt containing timing data
        output_file: Path to save the plot
    """
    thresholds = []
    apriori_times = []
    fp_times = []
    
    # Read results file
    with open(results_file, 'r') as f:
        lines = f.readlines()
        
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split(',')
            thresholds.append(float(parts[0]))
            apriori_times.append(float(parts[1]))
            fp_times.append(float(parts[2]))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, apriori_times, marker='o', label='Apriori', linewidth=2)
    plt.plot(thresholds, fp_times, marker='s', label='FP-Growth', linewidth=2)
    
    plt.xlabel('Support Threshold (%)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Runtime Comparison: Apriori vs FP-Growth', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 plot_results.py <results_file> <output_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2]
    
    plot_results(results_file, output_file)
