./env.sh

echo "========================================================"
echo "Running on Mutagenicity Dataset"
echo "========================================================"

DB_GRAPH="data/q3_datasets/Mutagenicity/graphs.txt"
QUERY_GRAPH="data/query_dataset/muta_final_visible"
PREFIX="muta"

# 1. Identify discriminative subgraphs
echo "[${PREFIX}] Identifying discriminative subgraphs..."
./identify.sh ${DB_GRAPH} ${PREFIX}_discriminative.pkl

# 2. Convert Database
echo "[${PREFIX}] Converting database graphs..."
./convert.sh ${DB_GRAPH} ${PREFIX}_discriminative.pkl ${PREFIX}_db_features.npy

# 3. Convert Queries
echo "[${PREFIX}] Converting query graphs..."
./convert.sh ${QUERY_GRAPH} ${PREFIX}_discriminative.pkl ${PREFIX}_query_features.npy

# 4. Generate Candidates
echo "[${PREFIX}] Generating candidates..."
./generate_candidates.sh ${PREFIX}_db_features.npy ${PREFIX}_query_features.npy ${PREFIX}_candidates.txt

# 5. Calculate Rq (Ground Truth)
echo "[${PREFIX}] Calculating ground truth (Rq)..."
python3 calculate_rq.py ${DB_GRAPH} ${QUERY_GRAPH} ${PREFIX}_rq.pkl

# 6. Calculate Score
echo "[${PREFIX}] Calculating score..."
python3 score_calculator.py ${PREFIX}_candidates.txt ${PREFIX}_rq.pkl


echo ""
echo "========================================================"
echo "Running on NCI-H23 Dataset"
echo "========================================================"

DB_GRAPH="data/q3_datasets/NCI-H23/graphs.txt"
QUERY_GRAPH="data/query_dataset/nci_final_visible"
PREFIX="nci"

# 1. Identify discriminative subgraphs
echo "[${PREFIX}] Identifying discriminative subgraphs..."
./identify.sh ${DB_GRAPH} ${PREFIX}_discriminative.pkl

# 2. Convert Database
echo "[${PREFIX}] Converting database graphs..."
./convert.sh ${DB_GRAPH} ${PREFIX}_discriminative.pkl ${PREFIX}_db_features.npy

# 3. Convert Queries
echo "[${PREFIX}] Converting query graphs..."
./convert.sh ${QUERY_GRAPH} ${PREFIX}_discriminative.pkl ${PREFIX}_query_features.npy

# 4. Generate Candidates
echo "[${PREFIX}] Generating candidates..."
./generate_candidates.sh ${PREFIX}_db_features.npy ${PREFIX}_query_features.npy ${PREFIX}_candidates.txt

# 5. Calculate Rq (Ground Truth)
echo "[${PREFIX}] Calculating ground truth (Rq)..."
python3 calculate_rq.py ${DB_GRAPH} ${QUERY_GRAPH} ${PREFIX}_rq.pkl

# 6. Calculate Score
echo "[${PREFIX}] Calculating score..."
python3 score_calculator.py ${PREFIX}_candidates.txt ${PREFIX}_rq.pkl