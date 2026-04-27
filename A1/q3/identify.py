import sys
import os
import pandas as pd
import rustworkx as rx
import pickle
import scipy.stats
from collections import Counter
from gspan_mining import gSpan

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

def gspan_to_rx(g):
    """
    Converts gspan_mining.graph.Graph to rustworkx.PyGraph
    """
    G = rx.PyGraph(multigraph=False)
    # Map gspan vids to rx indices
    vid_map = {}
    
    # Add nodes
    # g.vertices is a dict {vid: Vertex}
    # We want to maybe sort by vid to be deterministic?
    sorted_vids = sorted(g.vertices.keys())
    
    for vid in sorted_vids:
        v = g.vertices[vid]
        idx = G.add_node({'label': str(v.vlb)})
        vid_map[vid] = idx
        
    # Add edges
    for vid in sorted_vids:
        v = g.vertices[vid]
        for dst, e in v.edges.items():
            if vid < dst:
                # Add edge
                if dst in vid_map:
                    G.add_edge(vid_map[vid], vid_map[dst], {'label': str(e.elb)})
    return G


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
    
    db_v_dists = {}
    db_e_dists = {}
    for g in graphs:
        v_c = Counter()
        e_c = Counter()
        nodes = {}
        for line in g:
            parts = line.split()
            if parts[0] == 'v':
                lbl = parts[2]
                v_c[lbl] += 1
                nodes[parts[1]] = lbl
            elif parts[0] == 'e':
                u_lbl = nodes[parts[1]]
                v_lbl = nodes[parts[2]]
                e_lbl = parts[3]
                edge_tup = tuple(sorted([u_lbl, v_lbl])) + (e_lbl,)
                e_c[edge_tup] += 1
        for lbl, count in v_c.items():
            db_v_dists.setdefault(lbl, []).append(count)
        for lbl, count in e_c.items():
            db_e_dists.setdefault(lbl, []).append(count)
            
    def prob_feature_ge(dist_list, k, total_graphs):
        if not dist_list: return 0.0
        return sum(1 for x in dist_list if x >= k) / total_graphs
    
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
            sup = report.iloc[i]['support']
            g_rx = gspan_to_rx(p.to_graph())
            size = g_rx.num_edges() 
            
            if size not in subgraphs_by_size:
                subgraphs_by_size[size] = []
            subgraphs_by_size[size].append({
                'support': sup,
                'rx': g_rx, # Store as rx
                'gspan': p 
            })
            
    except Exception as e:
        print(f"Error processing support: {e}")
        return
    
    sizes = sorted(subgraphs_by_size.keys())
    
    # Helper for matching
    def node_match(n1, n2):
        return n1['label'] == n2['label']

    def edge_match(e1, e2):
        return e1['label'] == e2['label']
    
    def is_subgraph(small, big):
        return rx.is_subgraph_isomorphic(big, small, node_matcher=node_match, edge_matcher=edge_match, induced=False)

    def select_discriminative(subgraphs_by_size, gamma):
        print(f"Selecting discriminative subgraphs (gamma={gamma})...")
        selected = []
        
        # Process by size
        for size in sizes:
            candidates = subgraphs_by_size[size]
            # print(f"Processing {len(candidates)} subgraphs of size {size}...")
            
            for cand in candidates:
                g = cand['rx']
                p_gspan = cand['gspan']
                sup = cand['support']
                
                is_disc = 1
                
                # Check against parents of size size-1
                if size - 1 in subgraphs_by_size:
                    parents = subgraphs_by_size[size - 1]
                    for p in parents:
                        p_rx = p['rx']
                        p_sup = p['support']
                        
                        if sup >= gamma * p_sup:
                            if is_subgraph(p_rx, g):
                                is_disc = 0
                                break   
                
                if is_disc:
                    if size > 1:
                        v_c = Counter()
                        e_c = Counter()
                        for idx in g.node_indices():
                            v_c[g.get_node_data(idx)['label']] += 1
                        for u, v, e_data in g.weighted_edge_list():
                            u_lbl = g.get_node_data(u)['label']
                            v_lbl2 = g.get_node_data(v)['label']
                            e_lbl = e_data['label']
                            edge_tup = tuple(sorted([u_lbl, v_lbl2])) + (e_lbl,)
                            e_c[edge_tup] += 1
                        
                        P_g = 1.0
                        total_graphs = len(graphs)
                        for lbl, count in v_c.items():
                            P_g *= prob_feature_ge(db_v_dists.get(lbl, []), count, total_graphs)
                        for lbl, count in e_c.items():
                            P_g *= prob_feature_ge(db_e_dists.get(lbl, []), count, total_graphs)
                            
                        if P_g > 0:
                            score = scipy.stats.binom.sf(sup - 1, total_graphs, P_g)
                        else:
                            score = 0.0
                        selected.append((g, score))
        return selected

    # First try with gamma 0.8
    final_selected = select_discriminative(subgraphs_by_size, 0.8)
    
    # Check if we have enough
    if len(final_selected) < 50:
        print(f"Count {len(final_selected)} < 50. Retrying with gamma=0.9...")
        final_selected = select_discriminative(subgraphs_by_size, 0.9)
    
    # Global Median Support Selection
    print("Selecting Top 50 by Score...")
    
    # Sort by score (ascending, as score is distance from 50%)
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
