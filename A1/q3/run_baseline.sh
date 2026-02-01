./env.sh

echo "========================================================"
echo "Running Baseline (Top 50 Frequent) on Mutagenicity"
echo "========================================================"

DB_GRAPH="data/q3_datasets/Mutagenicity/graphs.txt"
QUERY_GRAPH="data/query_dataset/muta_final_visible"
PREFIX="muta_base"

# 1. Identify discriminative subgraphs (BASELINE)
echo "[${PREFIX}] Identifying BASELINE subgraphs..."
python3 identify_baseline.py ${DB_GRAPH} ${PREFIX}_discriminative.pkl

# 2. Convert Database
echo "[${PREFIX}] Converting database graphs..."
./convert.sh ${DB_GRAPH} ${PREFIX}_discriminative.pkl ${PREFIX}_db_features.npy

# 3. Convert Queries
echo "[${PREFIX}] Converting query graphs..."
./convert.sh ${QUERY_GRAPH} ${PREFIX}_discriminative.pkl ${PREFIX}_query_features.npy

# 4. Generate Candidates
echo "[${PREFIX}] Generating candidates..."
./generate_candidates.sh ${PREFIX}_db_features.npy ${PREFIX}_query_features.npy ${PREFIX}_candidates.txt

# 5. Calculate Rq (Ground Truth) -- REUSE EXISTING IF POSSIBLE TO SAVE TIME
RQ_FILE="muta_rq.pkl" 
if [ ! -f "$RQ_FILE" ]; then
    echo "[${PREFIX}] Calculating ground truth (Rq)..."
    python3 calculate_rq.py ${DB_GRAPH} ${QUERY_GRAPH} ${RQ_FILE}
else
    echo "[${PREFIX}] Reusing existing ground truth file: ${RQ_FILE}"
fi

# 6. Calculate Score
echo "[${PREFIX}] Calculating score..."
python3 score_calculator.py ${PREFIX}_candidates.txt ${RQ_FILE}
