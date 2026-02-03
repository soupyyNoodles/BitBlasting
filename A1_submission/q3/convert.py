import sys
import pickle
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from joblib import Parallel, delayed
import multiprocessing

def parse_graphs(file_path):
    graphs = []
    current_graph_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if current_graph_lines:
                    # Convert previous graph to NX
                    graphs.append(lines_to_nx(current_graph_lines))
                current_graph_lines = []
            else:
                current_graph_lines.append(line)
        if current_graph_lines:
            graphs.append(lines_to_nx(current_graph_lines))
    return graphs

def lines_to_nx(lines):
    G = nx.Graph()
    # lines: 'v id label', 'e src dst label'
    for line in lines:
        parts = line.split()
        if parts[0] == 'v':
            # v id label
            vid = int(parts[1])
            vlb = str(parts[2])
            G.add_node(vid, label=vlb)
        elif parts[0] == 'e':
            # e src dst label
            src = int(parts[1])
            dst = int(parts[2])
            elb = str(parts[3])
            G.add_edge(src, dst, label=elb)
    return G

# Helper function for parallel processing
def process_feature(subgraph, graphs, index, total):
    """
    Checks isomorphism of 'subgraph' against all 'graphs'.
    Returns a column vector (N,) of 0s and 1s.
    """
    nm = isomorphism.categorical_node_match("label", None)
    em = isomorphism.categorical_edge_match("label", None)
    
    # Progress Print
    if index % 10 == 0:
        print(f"Processing feature {index+1}/{total}...", flush=True)
        
    column = np.zeros(len(graphs), dtype=np.int8)
    
    for i, G in enumerate(graphs):
        GM = isomorphism.GraphMatcher(G, subgraph, node_match=nm, edge_match=em)
        if GM.subgraph_is_isomorphic():
            column[i] = 1
            
    return column

def main():
    if len(sys.argv) < 4:
        print("Usage: python convert.py <input_graphs> <discriminative_subgraphs> <output_features>")
        sys.exit(1)

    input_path = sys.argv[1]
    subgraphs_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Load subgraphs
    with open(subgraphs_path, 'rb') as f:
        subgraphs = pickle.load(f)
    print(f"Loaded {len(subgraphs)} subgraphs.", flush=True)
    
    # Load input graphs
    graphs = parse_graphs(input_path)
    print(f"Loaded {len(graphs)} input graphs.", flush=True)
    
    # Initialize feature matrix
    N = len(graphs)
    k = len(subgraphs)
    
    print(f"Starting parallel job for {k} features on {multiprocessing.cpu_count()} cores...", flush=True)
    
    # Parallel processing
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_feature)(sub, graphs, j, k) 
        for j, sub in enumerate(subgraphs)
    )
    
    # Combine columns into matrix (N, k)
    # results is list of k arrays of shape (N,)
    # Stack columns -> (k, N) -> Transpose -> (N, k)
    features = np.column_stack(results)
    
    np.save(output_path, features)
    print(f"Saved features to {output_path}. Shape: {features.shape}", flush=True)

if __name__ == "__main__":
    main()
