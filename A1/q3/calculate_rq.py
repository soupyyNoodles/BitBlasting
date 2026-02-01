import sys
import os
import pickle
import networkx as nx
from networkx.algorithms import isomorphism
import multiprocessing

def parse_graphs(file_path):
    """
    Parses the graph dataset.
    Returns a list of networkx.Graph objects.
    """
    graphs = []
    current_graph_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if current_graph_lines:
                    graphs.append(lines_to_nx(current_graph_lines))
                current_graph_lines = []
            else:
                current_graph_lines.append(line)
        if current_graph_lines:
            graphs.append(lines_to_nx(current_graph_lines))
    return graphs

def lines_to_nx(lines):
    G = nx.Graph()
    for line in lines:
        parts = line.split()
        if parts[0] == 'v':
            vid = int(parts[1])
            vlb = str(parts[2])
            G.add_node(vid, label=vlb)
        elif parts[0] == 'e':
            src = int(parts[1])
            dst = int(parts[2])
            elb = str(parts[3])
            G.add_edge(src, dst, label=elb)
    return G


# Global variable for worker processes
global_db_graphs = None

def init_worker(db_graphs):
    """
    Initializer for worker processes to set the global db_graphs.
    This avoids pickling the large list for every task.
    """
    global global_db_graphs
    global_db_graphs = db_graphs

def check_isomorphism(args):
    """
    Worker function to check isomorphism for a single query against all database graphs.
    """
    q_idx, query_graph = args # Removed db_graphs from args
    
    # Access global db_graphs
    db_graphs = global_db_graphs
    
    # Create matchers inside the worker to avoid pickling issues with local functions
    nm = isomorphism.categorical_node_match("label", None)
    em = isomorphism.categorical_edge_match("label", None)
    
    count = 0
    # For small queries and large DB, typically we check if query is subgraph of DB graph
    for db_graph in db_graphs:
        GM = isomorphism.GraphMatcher(db_graph, query_graph, node_match=nm, edge_match=em)
        if GM.subgraph_is_isomorphic():
            count += 1
    return q_idx, count

def main():
    if len(sys.argv) < 4:
        print("Usage: python calculate_rq.py <db_graphs_path> <query_graphs_path> <output_rq_pkl>")
        sys.exit(1)

    db_path = sys.argv[1]
    query_path = sys.argv[2]
    output_path = sys.argv[3]

    print(f"Loading database graphs from {db_path}...")
    db_graphs = parse_graphs(db_path)
    print(f"Loaded {len(db_graphs)} database graphs.")

    print(f"Loading query graphs from {query_path}...")
    query_graphs = parse_graphs(query_path)
    print(f"Loaded {len(query_graphs)} query graphs.")

    # Prepare arguments for multiprocessing
    # Only pass query index and query graph. db_graphs is passed via initializer.
    tasks = []
    for i, q in enumerate(query_graphs):
        tasks.append((i, q))

    print(f"Starting isomorphism checks for {len(query_graphs)} queries using multiprocessing...")
    
    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    results = {}
    
    # Pass db_graphs to initializer
    with multiprocessing.Pool(processes=num_cpus, initializer=init_worker, initargs=(db_graphs,)) as pool:
        # Use simple map or imap without tqdm
        for q_idx, count in pool.imap_unordered(check_isomorphism, tasks):
            results[q_idx] = count
            if len(results) % 10 == 0:
                print(f"Processed {len(results)}/{len(query_graphs)} queries...", flush=True)

    # Save results
    print(f"Saving results to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("Done.")

if __name__ == "__main__":
    main()
