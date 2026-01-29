env.sh -> sets up the environment
identify.py -> identifies 50 discriminative graphs from the dataset and saves them in discriminative.pkl
convert.py -> creates feature vector for the dataset and query graphs and saves them in db_features.npy and query_features.npy
generate_candidates.py -> generates candidates by comparing the feature vectors of query graph with graphs from dataset using broadcasting, saves them in generate_candidates.py
