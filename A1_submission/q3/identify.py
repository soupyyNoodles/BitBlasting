import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from gspan_mining import gSpan
from gspan_mining import gspan

# Patch gspan if needed (for pandas compatibility)
try:
    pd.DataFrame.append
except AttributeError:
    # Monkey patch fix for pandas 2.0+
    def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        # Implementation of append using concat
        from pandas import concat
        if isinstance(other, (list, tuple)):
            other = pd.DataFrame(other)
            if ignore_index:
                 pass
        
        if not isinstance(other, pd.DataFrame) and not isinstance(other, (list, tuple)):
             other = pd.DataFrame([other])
             
        return concat([self, other], ignore_index=ignore_index, sort=sort)
        
    pd.DataFrame.append = _append

def parse_graphs(file_path):
    """
    Parses the graph dataset.
    Returns a list of graphs, each is a list of lines (v and e).
    """
    graphs = []
    current_graph = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if current_graph:
                    graphs.append(current_graph)
                current_graph = []
            else:
                current_graph.append(line)
        if current_graph:
            graphs.append(current_graph)
    return graphs

def write_gspan_format(graphs, output_path):
    with open(output_path, 'w') as f:
        for i, graph in enumerate(graphs):
            f.write(f"t # {i}\n")
            for line in graph:
                f.write(f"{line}\n")
    return output_path

def gspan_to_nx(g):
    """
    Converts gspan_mining.graph.Graph to networkx.Graph
    """
    G = nx.Graph()
    for vid, v in g.vertices.items():
        G.add_node(vid, label=str(v.vlb))
    for vid, v in g.vertices.items():
        for dst, e in v.edges.items():
            if vid < dst:
                G.add_edge(vid, dst, label=str(e.elb))
    return G

def is_discriminative(subgraph_support, parent_supports, gamma=0.9):
    for p_sup in parent_supports:
        if subgraph_support >= gamma * p_sup:
            return False # Redundant
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python identify_gindex.py <input_graphs> <output_subgraphs> [min_support_pct]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    min_support_pct = 0.20
    if len(sys.argv) > 3:
        min_support_pct = float(sys.argv[3])
    
    print(f"Reading graphs from {input_path}...")
    graphs = parse_graphs(input_path)
    temp_gspan_file = "temp_gspan_input_gindex.data"
    write_gspan_format(graphs, temp_gspan_file)
    
    min_support = int(len(graphs) * min_support_pct)
    if min_support < 5: min_support = 5
    
    print(f"Mining with min_support={min_support} ({min_support_pct*100}%)...")
    
    gs = gSpan(
        database_file_name=temp_gspan_file,
        min_support=min_support,
        min_num_vertices=2,
        max_num_vertices=10,
        visualize=False
    )
    
    # Suppress gSpan internal printing
    import contextlib
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
             gs.run()
    
    print(f"Mined {len(gs._frequent_subgraphs)} subgraphs.")
    
    subgraphs_by_size = {}
    
    try:
        report = gs._report_df
        if len(report) != len(gs._frequent_subgraphs):
             print(f"Warning: report length {len(report)} != subgraphs length {len(gs._frequent_subgraphs)}")
        
        for i, p in enumerate(gs._frequent_subgraphs):
            sup = report.iloc[i]['support']
            g_nx = gspan_to_nx(p.to_graph())
            size = g_nx.number_of_edges() # gIndex usually uses edges or vertices. Edges is more standard for substructure.
            
            if size not in subgraphs_by_size:
                subgraphs_by_size[size] = []
            subgraphs_by_size[size].append({
                'support': sup,
                'nx': g_nx,
                'gspan': p # Keep if needed, though matching nx is safer for isomorphism
            })
            
    except Exception as e:
        print(f"Error processing support: {e}")
        return

    selected_subgraphs = [] # List of {'nx': g, 'support': s, 'id': i}
    
    sizes = sorted(subgraphs_by_size.keys())
    
    from networkx.algorithms.isomorphism import GraphMatcher
    
    def is_subgraph(small, big):
        gm = GraphMatcher(big, small, node_match=lambda n1,n2: n1['label']==n2['label'], edge_match=lambda e1,e2: e1['label']==e2['label'])
        return gm.subgraph_is_isomorphic()

    print("Selecting discriminative subgraphs (gIndex heuristic)...")
    
    final_selected = []
    
    # Process by size
    for size in sizes:
        candidates = subgraphs_by_size[size]
        print(f"Processing {len(candidates)} subgraphs of size {size}...")
        
        for cand in candidates:
            g = cand['nx']
            sup = cand['support']
            g.graph['support'] = sup
            
            is_disc = 1
            
            # Check against parents of size size-1
            if size - 1 in subgraphs_by_size:
                parents = subgraphs_by_size[size - 1]
                for p in parents:
                    p_nx = p['nx']
                    p_sup = p['support']
                    
                    gamma = 0.9
                    if sup >= gamma * p_sup:
                        if is_subgraph(p_nx, g):
                            # if abs(sup - len(graphs) * 0.5) < abs(p_sup - len(graphs) * 0.5):
                            #     if p_nx in final_selected:
                            #         final_selected.remove(p_nx)
                            #     is_disc = 2
                            # elif is_disc != 2: 
                            #     is_disc = 0
                            is_disc = 0
                            break   
            
            if is_disc:
                if size > 1:
                    score = abs(sup - len(graphs) * 0.5) / len(graphs)
                    final_selected.append((g, score))
    
    # Global Median Support Selection
    print("Selecting Top 50 by Score...")
    
    # Sort by score (ascending, as score is distance from 50%)
    # score = abs(sup - len(graphs) * 0.5) / len(graphs)
    # Lower score is better (closer to 50% split)
    
    final_selected.sort(key=lambda x: x[1])
    
    # Take top 50
    final_selected = [x[0] for x in final_selected[:50]]
    
    print(f"Selected {len(final_selected)} discriminative subgraphs.")
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_selected, f)
    
    if os.path.exists(temp_gspan_file):
        os.remove(temp_gspan_file)

if __name__ == "__main__":
    main()
