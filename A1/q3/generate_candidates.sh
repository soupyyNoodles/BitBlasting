#!/bin/bash

# generate_candidates.sh <path_database_graph_features> <path_query_graph_features> <path_out_file>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_database_graph_features> <path_query_graph_features> <path_out_file>"
    exit 1
fi

DB_FEATS=$1
QUERY_FEATS=$2
OUTPUT=$3

python3 generate_candidates.py "$DB_FEATS" "$QUERY_FEATS" "$OUTPUT"
