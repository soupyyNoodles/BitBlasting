#!/usr/bin/env python3
"""
Dataset format converter for frequent subgraph mining algorithms.
Converts Yeast dataset format to gSpan, Gaston, and FSG formats.
"""

import os
import sys


def parse_yeast_dataset(filepath):
    """
    Parse the Yeast dataset format.
    Format:
        #graphID
        number of nodes
        node labels (one per line)
        number of edges
        edges as: source_node_id, destination_node_id, edge_type
    
    Returns a list of graphs, where each graph is a dict with:
        - 'id': graph ID
        - 'nodes': list of node labels
        - 'edges': list of (src, dst, edge_label) tuples
    """
    graphs = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Graph ID line starts with #
        if line.startswith('#'):
            graph_id = line[1:]
            i += 1

            # Skip any empty lines after graph id
            while i < len(lines) and not lines[i].strip():
                i += 1

            # Number of nodes
            try:
                num_nodes = int(lines[i].strip())
            except (ValueError, IndexError):
                raise ValueError(
                    f"Expected integer num_nodes after graph id '#{graph_id}', got: "
                    f"{lines[i].strip() if i < len(lines) else 'EOF'}\n"
                    "Check the dataset file path. If you passed 'actives.txt', "
                    "use 'Yeast/167.txt_graph' instead."
                )
            i += 1
            
            # Read node labels
            nodes = []
            for _ in range(num_nodes):
                node_label = lines[i].strip()
                nodes.append(node_label)
                i += 1
            
            # Number of edges
            num_edges = int(lines[i].strip())
            i += 1
            
            # Read edges
            edges = []
            for _ in range(num_edges):
                edge_parts = lines[i].strip().split()
                src = int(edge_parts[0])
                dst = int(edge_parts[1])
                edge_label = int(edge_parts[2])
                edges.append((src, dst, edge_label))
                i += 1
            
            graphs.append({
                'id': graph_id,
                'nodes': nodes,
                'edges': edges
            })
        else:
            i += 1
    
    return graphs


def create_label_mapping(graphs):
    """
    Create mappings from string labels to integer labels.
    Returns (node_label_map, edge_label_map)
    """
    node_labels = set()
    edge_labels = set()
    
    for graph in graphs:
        for label in graph['nodes']:
            node_labels.add(label)
        for _, _, edge_label in graph['edges']:
            edge_labels.add(edge_label)
    
    # Sort for consistent mapping
    node_label_map = {label: idx for idx, label in enumerate(sorted(node_labels))}
    edge_label_map = {label: idx for idx, label in enumerate(sorted(edge_labels))}
    
    return node_label_map, edge_label_map


def convert_to_gspan_format(graphs, output_path, node_label_map=None):
    """
    Convert to gSpan/Gaston format:
        t # N
        v node_id label
        e src dst edge_label
    """
    with open(output_path, 'w') as f:
        for idx, graph in enumerate(graphs):
            f.write(f"t # {idx}\n")
            
            for node_id, label in enumerate(graph['nodes']):
                if node_label_map:
                    label_int = node_label_map[label]
                else:
                    label_int = label
                f.write(f"v {node_id} {label_int}\n")
            
            for src, dst, edge_label in graph['edges']:
                f.write(f"e {src} {dst} {edge_label}\n")


def convert_to_fsg_format(graphs, output_path, node_label_map=None):
    """
    Convert to FSG format:
        t # graph_id
        v node_id label
        u src dst edge_label
    Note: FSG uses 'u' for undirected edges
    """
    with open(output_path, 'w') as f:
        for idx, graph in enumerate(graphs):
            f.write(f"t # {idx}\n")
            
            for node_id, label in enumerate(graph['nodes']):
                if node_label_map:
                    label_int = node_label_map[label]
                else:
                    label_int = label
                f.write(f"v {node_id} {label_int}\n")
            
            for src, dst, edge_label in graph['edges']:
                f.write(f"u {src} {dst} {edge_label}\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_dataset.py <input_yeast_dataset> <output_dir>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Parsing Yeast dataset from {input_path}...")
    graphs = parse_yeast_dataset(input_path)
    print(f"Found {len(graphs)} graphs")
    
    # Create label mappings
    node_label_map, edge_label_map = create_label_mapping(graphs)
    print(f"Found {len(node_label_map)} unique node labels")
    print(f"Found {len(edge_label_map)} unique edge labels")
    
    # Save label mappings for reference
    with open(os.path.join(output_dir, 'node_labels.txt'), 'w') as f:
        for label, idx in sorted(node_label_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{label}\n")
    
    # Convert to different formats
    gspan_path = os.path.join(output_dir, 'yeast_gspan.txt')
    gaston_path = os.path.join(output_dir, 'yeast_gaston.txt')
    fsg_path = os.path.join(output_dir, 'yeast_fsg.txt')
    
    print(f"Converting to gSpan format: {gspan_path}")
    convert_to_gspan_format(graphs, gspan_path, node_label_map)
    
    print(f"Converting to Gaston format: {gaston_path}")
    # Gaston format is the same as gSpan
    convert_to_gspan_format(graphs, gaston_path, node_label_map)
    
    print(f"Converting to FSG format: {fsg_path}")
    convert_to_fsg_format(graphs, fsg_path, node_label_map)
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()
