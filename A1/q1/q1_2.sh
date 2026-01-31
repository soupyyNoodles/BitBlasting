#!/bin/bash

# Usage: bash q1_2.sh <universal_itemset> <num_transactions>
# This script generates a dataset and runs Task 1 experiments on it.

if [ "$#" -ne 2 ]; then
    echo "Usage: bash q1_2.sh <universal_itemset> <num_transactions>"
    exit 1
fi

UNIVERSAL_ITEMSET="$1"
NUM_TRANSACTIONS="$2"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generate the dataset
echo "Generating dataset with $UNIVERSAL_ITEMSET items and $NUM_TRANSACTIONS transactions..."
python3 "$SCRIPT_DIR/generate_transactions.py" "$UNIVERSAL_ITEMSET" "$NUM_TRANSACTIONS" --output "$SCRIPT_DIR/generated_transactions.dat"

if [ $? -ne 0 ]; then
    echo "Error: Dataset generation failed"
    exit 1
fi

echo ""
echo "Dataset generated: $SCRIPT_DIR/generated_transactions.dat"
echo "To run experiments on this dataset, use:"
echo "  bash q1_1.sh <path_apriori_executable> <path_fp_executable> $SCRIPT_DIR/generated_transactions.dat <path_out>"
