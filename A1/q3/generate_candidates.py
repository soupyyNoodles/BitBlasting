import sys
import numpy as np

#TODO: Read up this piece of code again

def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_candidates.py <db_features> <query_features> <output_file>")
        sys.exit(1)

    db_path = sys.argv[1]
    query_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Load features (.npy)
    print("Loading features...")
    db_feats = np.load(db_path)
    query_feats = np.load(query_path)
    
    # db_feats: N x k
    # query_feats: M x k
    
    N, k1 = db_feats.shape
    M, k2 = query_feats.shape
    
    if k1 != k2:
        print(f"Error: Feature dimensions mismatch! DB: {k1}, Query: {k2}")
        sys.exit(1)
        
    print(f"Generating candidates for {M} queries against {N} DB graphs...")
    
    # Logic:
    # if query has feature f (1), DB graph MUST have feature f (1).
    # if query has 0, DB graph can be 0 or 1.
    # Condition: NOT (Q[i]==1 AND G[i]==0)
    # Equivalent to: (Q[i] <= G[i]) for all i where Q[i]=1.
    
    # Vectorized check:
    # Using broadcasting is expensive if matrices are huge (N*M*k).
    # But usually manageable.
    # N ~ 1000-4000
    # M ~ 100
    # k ~ 50
    # 4000 * 100 * 50 = 20M elems. fits in memory easily.
    
    # Expand dims
    # query: M x 1 x k
    # db:    1 x N x k
    
    Q_broad = query_feats[:, np.newaxis, :]
    DB_broad = db_feats[np.newaxis, :, :]
    
    # Check invalid: Q=1 and DB=0
    invalid_mask = (Q_broad == 1) & (DB_broad == 0)
    
    # Candidate if NO invalid feature match
    # invalid_mask: M x N x k
    # any invalid feature?
    is_invalid = np.any(invalid_mask, axis=2) # M x N
    
    # write results
    with open(output_path, 'w') as f:
        for q_idx in range(M):
            # indices where is_invalid is False
            candidates = np.where(~is_invalid[q_idx])[0]
            
            f.write(f"q # {q_idx}\n")
            c_str = " ".join(map(str, candidates))
            f.write(f"c # {c_str}\n")

    print(f"Candidates written to {output_path}")

if __name__ == "__main__":
    main()
