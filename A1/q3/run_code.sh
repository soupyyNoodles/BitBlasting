./env.sh

./identify.sh data/q3_datasets/Mutagenicity/graphs.txt discriminative.pkl

# Convert Database
./convert.sh data/q3_datasets/Mutagenicity/graphs.txt discriminative.pkl db_features.npy

# Convert Queries
./convert.sh data/query_dataset/muta_final_visible discriminative.pkl query_features.npy

./generate_candidates.sh db_features.npy query_features.npy candidates.txt