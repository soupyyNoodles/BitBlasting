import sys
import os
import pickle
import rustworkx as rx
# import networkx as nx
# from networkx.algorithms import isomorphism
import multiprocessing

def parse_graphs(file_path):
    """
    Parses the graph dataset.
    Returns a list of rustworkx.PyGraph objects.
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
                    graphs.append(lines_to_rx(current_graph_lines))
                current_graph_lines = []
            else:
                current_graph_lines.append(line)
        if current_graph_lines:
            graphs.append(lines_to_rx(current_graph_lines))
    return graphs

def lines_to_rx(lines):
    G = rx.PyGraph(multigraph=False)
    vid_map = {} # Map file VID to rx index
    
    for line in lines:
        parts = line.split()
        if parts[0] == 'v':
            vid = int(parts[1])
            vlb = str(parts[2])
            idx = G.add_node({'label': vlb})
            vid_map[vid] = idx
        elif parts[0] == 'e':
            src = int(parts[1])
            dst = int(parts[2])
            elb = str(parts[3])
            if src in vid_map and dst in vid_map:
                G.add_edge(vid_map[src], vid_map[dst], {'label': elb})
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

def check_query(args):
    q_idx, query_graph = args
    db_graphs = global_db_graphs
    
    def node_match(n1, n2):
        return n1['label'] == n2['label']

    def edge_match(e1, e2):
        return e1['label'] == e2['label']
    
    count = 0
    for i, db_graph in enumerate(db_graphs):
        # Check if query_graph is a subgraph of db_graph
        # rx.is_subgraph_isomorphic(big, small, ...)
        if rx.is_subgraph_isomorphic(db_graph, query_graph, node_matcher=node_match, edge_matcher=edge_match, induced=False):
            count += 1
            
    return q_idx, count

def main():
    if len(sys.argv) < 4:
        print("Usage: python calculate_rq.py <db_file> <query_file> <output_rq_pkl>")
        sys.exit(1)

    db_path = sys.argv[1]
    query_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print(f"Loading database graphs from {db_path}...")
    db_graphs = parse_graphs(db_path)
    print(f"Loaded {len(db_graphs)} database graphs.")
    
    print(f"Loading query graphs from {query_path}...")
    queries = parse_graphs(query_path)
    print(f"Loaded {len(queries)} query graphs.")
    
    # Setup global
    global global_db_graphs
    global_db_graphs = db_graphs
    
    queries_with_idx = list(enumerate(queries))
    
    print(f"Starting isomorphism checks for {len(queries)} queries using multiprocessing...")
    
    rq_counts = {}
    
    # Use fewer processes to avoid OOM if graphs are large, but usually cpu_count is fine
    # Chunksize 1 for better progress updates
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for i, (q_idx, count) in enumerate(pool.imap_unordered(check_query, queries_with_idx)):
            rq_counts[q_idx] = count
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(queries)} queries...", flush=True)

    print(f"Saving results to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(rq_counts, f)
    print("Done.")

if __name__ == "__main__":
    main()
