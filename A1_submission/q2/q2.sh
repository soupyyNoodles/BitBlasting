#!/bin/bash

# q2.sh - Run frequent subgraph mining experiments
# Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>

if [ "$#" -ne 5 ]; then
    echo "Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>"
    exit 1
fi

GSPAN_EXEC="$1"
FSG_EXEC="$2"
GASTON_EXEC="$3"
DATASET_PATH="$4"
OUTPUT_DIR="$5"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the experiments
python3 "$SCRIPT_DIR/run_experiments.py" "$GSPAN_EXEC" "$FSG_EXEC" "$GASTON_EXEC" "$DATASET_PATH" "$OUTPUT_DIR"

echo "Experiments complete. Output saved to $OUTPUT_DIR"
