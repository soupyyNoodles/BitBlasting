#!/bin/bash

# identify.sh <path_graph_dataset> <path_discriminative_subgraphs>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_graph_dataset> <path_discriminative_subgraphs>"
    exit 1
fi

DATASET=$1
OUTPUT=$2

python3 identify.py "$DATASET" "$OUTPUT"
