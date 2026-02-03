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

if [ ! -x "$APRIORI_EXEC" ]; then
    echo "Error: Apriori executable not found or not executable: $APRIORI_EXEC"
    exit 1
fi

if [ ! -x "$FP_EXEC" ]; then
    echo "Error: FP-Growth executable not found or not executable: $FP_EXEC"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found: $DATASET"
    exit 1
fi

# Create output directory and logs subdirectory if they don't exist
mkdir -p "$OUTPUT_DIR"
LOGS_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGS_DIR"

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
    
    timeout 3600 "$APRIORI_EXEC" -s${threshold} "$DATASET" "$OUTPUT_DIR/ap${threshold}" > "$LOGS_DIR/ap${threshold}.log" 2>&1
    STATUS=$?
    if [ $STATUS -eq 124 ]; then
        echo "  Apriori ${threshold}%: Timed out after 3600 seconds - generating empty output"
        touch "$OUTPUT_DIR/ap${threshold}"
    elif [ $STATUS -ne 0 ]; then
        if grep -q "no (frequent) items found" "$LOGS_DIR/ap${threshold}.log"; then
            echo "  Apriori ${threshold}%: No frequent items found"
        else
            echo "  Warning: Apriori failed at ${threshold}% (see $LOGS_DIR/ap${threshold}.log) - continuing with next threshold"
            touch "$OUTPUT_DIR/ap${threshold}"
        fi
    fi
    
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
    
    timeout 3600 "$FP_EXEC" -s${threshold} "$DATASET" "$OUTPUT_DIR/fp${threshold}" > "$LOGS_DIR/fp${threshold}.log" 2>&1
    STATUS=$?
    if [ $STATUS -eq 124 ]; then
        echo "  FP-Growth ${threshold}%: Timed out after 3600 seconds - generating empty output"
        touch "$OUTPUT_DIR/fp${threshold}"
    elif [ $STATUS -ne 0 ]; then
        if grep -q "no (frequent) items found" "$LOGS_DIR/fp${threshold}.log"; then
            echo "  FP-Growth ${threshold}%: No frequent items found"
        else
            echo "  Warning: FP-Growth failed at ${threshold}% (see $LOGS_DIR/fp${threshold}.log) - continuing with next threshold"
            touch "$OUTPUT_DIR/fp${threshold}"
        fi
    fi
    
    END=$(python3 -c 'import time; print(time.time())')
    RUNTIME=$(python3 -c "print($END - $START)")
    
    FP_TIMES+=("$RUNTIME")
    echo "  FP-Growth ${threshold}%: ${RUNTIME}s"
done

# Save timing results to a file in logs directory for Python plotting script
RESULTS_FILE="$LOGS_DIR/results.txt"
echo "thresholds,apriori_times,fp_times" > "$RESULTS_FILE"

for i in "${!THRESHOLDS[@]}"; do
    echo "${THRESHOLDS[$i]},${APRIORI_TIMES[$i]},${FP_TIMES[$i]}" >> "$RESULTS_FILE"
done

# Get the directory of this script to find plot_results.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generate the plot using Python
echo "Generating plot..."
python3 "$SCRIPT_DIR/plot_results.py" "$RESULTS_FILE" "$OUTPUT_DIR/plot.png"

echo ""
echo "Execution complete!"
echo "Output directory: $OUTPUT_DIR"
echo "  - Algorithm outputs: ap5, ap10, ap25, ap50, ap90, fp5, fp10, fp25, fp50, fp90"
echo "  - Plot: plot.png"
echo "  - Logs: logs/"

echo "Done! Results saved to $OUTPUT_DIR"
