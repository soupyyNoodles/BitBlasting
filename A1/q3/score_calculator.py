import sys
import pickle
import numpy as np

def parse_candidates(file_path):
    """
    Parses candidates.dat file.
    Returns a dictionary {query_id (int): candidate_count (int)}
    """
    candidates_count = {}
    current_query_id = -1
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if parts[0] == 'q':
                # q # <id>
                current_query_id = int(parts[2])
            elif parts[0] == 'c':
                # c # <id1> <id2> ...
                # or just c # if empty
                if len(parts) > 2:
                    count = len(parts) - 2 # subtract 'c' and '#'
                else:
                    count = 0
                
                if current_query_id != -1:
                    candidates_count[current_query_id] = count
                    current_query_id = -1 # Reset
            
    return candidates_count

def main():
    if len(sys.argv) < 3:
        print("Usage: python score_calculator.py <candidates_dat_path> <rq_pkl_path>")
        sys.exit(1)

    candidates_path = sys.argv[1]
    rq_path = sys.argv[2]
    
    print(f"Reading candidates from {candidates_path}...")
    candidates_counts = parse_candidates(candidates_path)
    print(f"Loaded candidate counts for {len(candidates_counts)} queries.")
    
    print(f"Reading ground truth Rq from {rq_path}...")
    with open(rq_path, 'rb') as f:
        rq_counts = pickle.load(f)
    print(f"Loaded Rq counts for {len(rq_counts)} queries.")
    
    # Calculate scores
    scores = []
    
    # Assuming query IDs correspond to indices 0..N-1 or are consistent
    # The assignment example says "q # 1", "q # 2". Let's assume 1-based or whatever is in the file.
    # calculate_rq.py used 0-based index from the list of queries. 
    # parse_graphs usually returns list, so query 0 is the first graph.
    # We need to ensure alignment.
    # IF the input queries file has sequential queries, then query index i corresponds to q # (i) or (i+1).
    # Let's inspect candidates.dat later to be sure of the ID format. 
    # Usually students output "q # 0" or "q # 1". 
    # Let's align by intersection of keys if possible, or just iterate if keys match.
    
    common_ids = sorted(list(set(candidates_counts.keys()) & set(rq_counts.keys())))
    
    if not common_ids:
        print("Warning: No common query IDs found between candidates and Rq file.")
        print(f"Candidate keys sample: {list(candidates_counts.keys())[:5]}")
        print(f"Rq keys sample: {list(rq_counts.keys())[:5]}")
        # Try adjusting Rq keys if they are 0-based and candidates are 0-based or 1-based
        # If Rq keys are 0..N-1, and Candidates are 0..N-1, we are good.
        pass

    print(f"Calculating scores for {len(common_ids)} queries...")
    
    total_score = 0
    
    print(f"{'Query ID':<10} | {'|Rq|':<10} | {'|Cq|':<10} | {'Score':<10}")
    print("-" * 50)
    
    for qid in common_ids:
        rq = rq_counts[qid]
        cq = candidates_counts[qid]
        
        if cq == 0:
            if rq == 0:
                score = 1.0 # Perfect filtering (none exist, none returned) - arguably 1
            else:
                score = 0.0 # Candidates missed Rq! This is bad. Violation of Cq ⊇ Rq
                print(f"CRITICAL ERROR: q{qid} has |Rq|={rq} but |Cq|=0. Invalid candidate set.")
        else:
            score = rq / cq
            
        scores.append(score)
        total_score += score
        
        # Print first few or all? Let's print first 10 and then summary
        if len(scores) <= 50:
             print(f"{qid:<10} | {rq:<10} | {cq:<10} | {score:.4f}")

    if len(scores) > 0:
        avg_score = total_score / len(scores)
        print("-" * 50)
        print(f"Average Score: {avg_score:.4f}")
    else:
        print("No scores calculated.")

if __name__ == "__main__":
    main()
