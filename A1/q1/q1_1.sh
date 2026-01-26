#!/bin/bash

# Usage: bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>

if [ "$#" -ne 4 ]; then
    echo "Usage: bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>"
    exit 1
fi

APRIORI_EXEC="$1"
FP_EXEC="$2"
DATASET="$3"
OUTPUT_DIR="$4"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Support thresholds (in percentage)
THRESHOLDS=(5 10 25 50 90)

# Arrays to store runtimes
declare -a APRIORI_TIMES
declare -a FP_TIMES

echo "Running Apriori and FP-Growth at different support thresholds..."

# Run Apriori at different thresholds
for threshold in "${THRESHOLDS[@]}"; do
    echo "Running Apriori at ${threshold}% support..."
    
    # Time the execution
    START=$(python3 -c 'import time; print(time.time())')
    
    "$APRIORI_EXEC" -s-${threshold} "$DATASET" "$OUTPUT_DIR/ap${threshold}" > /dev/null 2>&1
    
    END=$(python3 -c 'import time; print(time.time())')
    RUNTIME=$(python3 -c "print($END - $START)")
    
    APRIORI_TIMES+=("$RUNTIME")
    echo "  Apriori ${threshold}%: ${RUNTIME}s"
done

# Run FP-Growth at different thresholds
for threshold in "${THRESHOLDS[@]}"; do
    echo "Running FP-Growth at ${threshold}% support..."
    
    # Time the execution
    START=$(python3 -c 'import time; print(time.time())')
    
    "$FP_EXEC" -s-${threshold} "$DATASET" "$OUTPUT_DIR/fp${threshold}" > /dev/null 2>&1
    
    END=$(python3 -c 'import time; print(time.time())')
    RUNTIME=$(python3 -c "print($END - $START)")
    
    FP_TIMES+=("$RUNTIME")
    echo "  FP-Growth ${threshold}%: ${RUNTIME}s"
done

# Save timing results to a file for Python plotting script
RESULTS_FILE="$OUTPUT_DIR/results.txt"
echo "thresholds,apriori_times,fp_times" > "$RESULTS_FILE"

for i in "${!THRESHOLDS[@]}"; do
    echo "${THRESHOLDS[$i]},${APRIORI_TIMES[$i]},${FP_TIMES[$i]}" >> "$RESULTS_FILE"
done

# Generate the plot using Python
echo "Generating plot..."
python3 plot_results.py "$RESULTS_FILE" "$OUTPUT_DIR/plot.png"

echo "Done! Results saved to $OUTPUT_DIR"
