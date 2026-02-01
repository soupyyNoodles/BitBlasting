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
                # If ignoring index, reset index of other to match self's range to avoid collision if desired
                # But concat with ignore_index=True handles it.
                pass
        
        # In a list context, other might be a single Series or dict, but gspan uses it to append a DF or Series.
        # Simplest shim:
        if not isinstance(other, pd.DataFrame) and not isinstance(other, (list, tuple)):
             # assume series or dict
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

def main():
    if len(sys.argv) < 3:
        print("Usage: python identify_baseline.py <input_graphs> <output_subgraphs>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Reading graphs from {input_path}...")
    graphs = parse_graphs(input_path)
    temp_gspan_file = "temp_gspan_input_baseline.data"
    write_gspan_format(graphs, temp_gspan_file)
    
    min_support = int(len(graphs) * 0.25)
    if min_support < 5: min_support = 5
    
    print(f"Mining with min_support={min_support}...")
    
    gs = gSpan(
        database_file_name=temp_gspan_file,
        min_support=min_support,
        min_num_vertices=2,
        max_num_vertices=10, 
        visualize=False
    )
    
    gs.run()
    
    print(f"Mined {len(gs._frequent_subgraphs)} subgraphs.")
    
    subgraphs_with_support = []
    
    try:
        report = gs._report_df
        if len(report) != len(gs._frequent_subgraphs):
             print(f"Warning: report length {len(report)} != subgraphs length {len(gs._frequent_subgraphs)}")
        
        for i, p in enumerate(gs._frequent_subgraphs):
            
            # The support is in report.iloc[i]['support']
            sup = report.iloc[i]['support']
            g_nx = gspan_to_nx(p.to_graph())
            subgraphs_with_support.append((sup, g_nx))
            
    except Exception as e:
        print(f"Error processing support: {e}")
        # Fallback if report fails
        subgraphs_with_support = [(0, gspan_to_nx(p.to_graph())) for p in gs._frequent_subgraphs]
    
    # BASELINE IMPLEMENTATION CHANGE:
    # Instead of discriminative score, sort purely by SUPPORT (descending)
    
    # Sort by support descending
    subgraphs_with_support.sort(key=lambda x: x[0], reverse=True)
    
    k = 50
    selected = [x[1] for x in subgraphs_with_support[:k]]
    
    print(f"Selected {len(selected)} baseline subgraphs (top frequents).")
    
    with open(output_path, 'wb') as f:
        pickle.dump(selected, f)
    
    if os.path.exists(temp_gspan_file):
        os.remove(temp_gspan_file)

if __name__ == "__main__":
    main()
