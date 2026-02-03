#!/bin/bash
set -e

# Configuration
MIN_SUP=0.20 # As per instructions
HEURISTIC_SCRIPT="identify.py"
CONVERT_SCRIPT="convert.py"
CANDIDATE_SCRIPT="generate_candidates.py"
SCORE_SCRIPT="score_calculator.py"
RQ_SCRIPT="calculate_rq.py"

# ---- Helper Function ----
run_pipeline() {
    DATASET_NAME=$1
    DB_PATH=$2
    QUERY_PATH=$3
    OUTPUT_PREFIX=$4
    
    echo "=========================================================="
    echo "Running Pipeline for: $DATASET_NAME"
    echo "Database: $DB_PATH"
    echo "Queries:  $QUERY_PATH"
    echo "Support:  $MIN_SUP"
    echo "=========================================================="

    # Derived Paths
    RQ_FILE="${OUTPUT_PREFIX}_rq.pkl"
    DISC_PKL="${OUTPUT_PREFIX}_discriminative.pkl"
    DB_FEAT="${OUTPUT_PREFIX}_db_features.npy"
    QUERY_FEAT="${OUTPUT_PREFIX}_query_features.npy"
    CANDIDATES="${OUTPUT_PREFIX}_candidates.txt"
    
    # 0. Pre-calculate Rq Sets (Not timed, ground truth)
    if [ ! -f "$RQ_FILE" ]; then
        echo "Calculating Rq Sets (Ground Truth)..."
        python3 "$RQ_SCRIPT" "$DB_PATH" "$QUERY_PATH" "$RQ_FILE"
    else
        echo "Rq Sets found ($RQ_FILE), skipping calculation."
    fi

    # 1. Timed Execution Block (Identification + Conversion + Generation)
    echo "Starting TIMED execution..."
    START_TIME=$(date +%s.%N)
    
    # A. Identification
    echo " -> Identifying discriminative subgraphs..."
    python3 "$HEURISTIC_SCRIPT" "$DB_PATH" "$DISC_PKL" "$MIN_SUP"
    
    # B. Conversion
    echo " -> Generating Database Features..."
    python3 "$CONVERT_SCRIPT" "$DB_PATH" "$DISC_PKL" "$DB_FEAT"
    
    echo " -> Generating Query Features..."
    python3 "$CONVERT_SCRIPT" "$QUERY_PATH" "$DISC_PKL" "$QUERY_FEAT"
    
    # C. Candiates
    echo " -> Generating Candidates..."
    python3 "$CANDIDATE_SCRIPT" "$DB_FEAT" "$QUERY_FEAT" "$CANDIDATES"
    
    END_TIME=$(date +%s.%N)
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    echo "TIMED execution finished."
    echo "Runtime (Identify + Convert + Generate): $RUNTIME seconds"
    
    # 2. Scoring (Not timed)
    echo "Calculating Score..."
    python3 "$SCORE_SCRIPT" "$CANDIDATES" "$RQ_FILE"
    
    echo "Pipeline for $DATASET_NAME completed."
    echo ""
}

# ---- Mutagenicity ----
MUTA_DB="data/q3_datasets/Mutagenicity/graphs.txt"
MUTA_QUERY="data/query_dataset/muta_final_visible"
MUTA_OUT="output_muta"

run_pipeline "Mutagenicity" "$MUTA_DB" "$MUTA_QUERY" "$MUTA_OUT"

# ---- NCI-H23 ----
# Use full graph set
NCI_DB="data/q3_datasets/NCI-H23/graphs.txt"
NCI_QUERY="data/query_dataset/nci_final_visible"
NCI_OUT="output_nci"

run_pipeline "NCI-H23" "$NCI_DB" "$NCI_QUERY" "$NCI_OUT"
