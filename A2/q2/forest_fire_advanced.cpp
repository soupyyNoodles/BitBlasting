#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <queue>
#include <random>
#include <algorithm>
#include <chrono>
#include <unordered_map>

using namespace std;

struct Edge {
    int src;
    int dst;
    double p;
    int id; // index in raw_edges
};

struct OriginalEdge {
    int u, v; double p;
    long long orig_u, orig_v;
};

// ==========================================
// Lengauer-Tarjan Algorithm Implementation
// ==========================================
int dfn_timer = 0;
vector<int> dfn, vertex, sdom, idom, dsu, best_node, parent_node;
vector<vector<int>> bucket;
vector<vector<int>> dominator_tree;

void init_lt(int num_nodes) {
    dfn_timer = 0;
    dfn.assign(num_nodes, 0);
    vertex.assign(num_nodes + 1, 0);
    sdom.assign(num_nodes, 0);
    idom.assign(num_nodes, 0);
    dsu.assign(num_nodes, 0);
    best_node.assign(num_nodes, 0);
    parent_node.assign(num_nodes, 0);
    bucket.assign(num_nodes, vector<int>());
    dominator_tree.assign(num_nodes, vector<int>());
    
    for (int i = 0; i < num_nodes; ++i) {
        dsu[i] = best_node[i] = sdom[i] = i;
    }
}

void dfs_lt(int u, const vector<vector<int>>& adj) {
    dfn[u] = ++dfn_timer;
    vertex[dfn_timer] = u;
    for (int v : adj[u]) {
        if (!dfn[v]) {
            parent_node[v] = u;
            dfs_lt(v, adj);
        }
    }
}

int eval(int u) {
    if (dsu[u] == u) return u;
    int res = eval(dsu[u]);
    if (dfn[sdom[best_node[dsu[u]]]] < dfn[sdom[best_node[u]]]) {
        best_node[u] = best_node[dsu[u]];
    }
    dsu[u] = res;
    return best_node[u];
}

void build_dominator_tree(int root, int num_nodes, const vector<vector<int>>& adj, const vector<vector<int>>& rev_adj) {
    init_lt(num_nodes);
    dfs_lt(root, adj);
    
    for (int i = dfn_timer; i >= 2; --i) {
        int u = vertex[i];
        for (int v : rev_adj[u]) {
            if (dfn[v] > 0) { // only consider reachable backwards
                int evaluated = eval(v);
                if (dfn[sdom[evaluated]] < dfn[sdom[u]]) {
                    sdom[u] = sdom[evaluated];
                }
            }
        }
        bucket[sdom[u]].push_back(u);
        int p = parent_node[u];
        dsu[u] = p; // link u to parent p
        
        for (int w : bucket[p]) {
            int evaluated = eval(w);
            if (sdom[evaluated] == sdom[w]) idom[w] = sdom[w];
            else idom[w] = evaluated;
        }
        bucket[p].clear();
    }
    for (int i = 2; i <= dfn_timer; ++i) {
        int u = vertex[i];
        if (idom[u] != sdom[u]) idom[u] = idom[idom[u]];
    }
    
    for (int i = 2; i <= dfn_timer; ++i) {
        int u = vertex[i];
        dominator_tree[idom[u]].push_back(u);
    }
}

// Compute subtree sizes recursively (only counting original graph nodes, ignoring dummy edge-nodes)
void compute_subtree_sizes(int u, const vector<vector<int>>& dom_tree, const vector<bool>& is_dummy, vector<int>& subtree_size) {
    subtree_size[u] = is_dummy[u] ? 0 : 1;
    for (int v : dom_tree[u]) {
        compute_subtree_sizes(v, dom_tree, is_dummy, subtree_size);
        subtree_size[u] += subtree_size[v];
    }
}


int main(int argc, char* argv[]) {
    auto start_time = chrono::steady_clock::now();

    if (argc < 7) {
        cerr << "Usage: <graph_file> <seed_file> <output_file> <k> <n_random_instances> <hops>" << endl;
        return 1;
    }
    string graph_path = argv[1];
    string seed_path = argv[2];
    string output_path = argv[3];
    int k = stoi(argv[4]);
    int n_random_instances = stoi(argv[5]);
    int hops = stoi(argv[6]);

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    unordered_map<long long, int> id_map;
    vector<long long> orig_id;
    auto get_id = [&](long long x) {
        if (id_map.find(x) == id_map.end()) {
            id_map[x] = orig_id.size();
            orig_id.push_back(x);
        }
        return id_map[x];
    };

    vector<OriginalEdge> raw_edges;
    ifstream fg(graph_path);
    long long ou, ov;
    double p;
    while (fg >> ou >> ov >> p) {
        int u = get_id(ou);
        int v = get_id(ov);
        raw_edges.push_back({u, v, p, ou, ov});
    }
    fg.close();
    
    vector<int> initial_seeds;
    ifstream fs(seed_path);
    long long s;
    while (fs >> s) {
        if (id_map.find(s) != id_map.end()) {
            initial_seeds.push_back(id_map[s]);
        }
    }
    fs.close();
    
    // Create a unified super-source
    int n_orig = orig_id.size();
    int super_source = n_orig;
    int n = n_orig + 1;
    int num_edges = raw_edges.size();
    
    vector<vector<Edge>> adj_orig(n);
    for (int i = 0; i < num_edges; ++i) {
        adj_orig[raw_edges[i].u].push_back({raw_edges[i].u, raw_edges[i].v, raw_edges[i].p, i});
    }
    for (int seed : initial_seeds) {
        adj_orig[super_source].push_back({super_source, seed, 1.0, -1}); // deterministic edge
    }

    vector<bool> edge_blocked(num_edges, false);
    vector<int> blocked_edges_ids;

    mt19937 rng(42);
    uniform_real_distribution<double> dist(0.0, 1.0);

    int num_sims = max(n_random_instances, 50);

    for (int step = 0; step < k; ++step) {
        auto current_time = chrono::steady_clock::now();
        double elapsed_sec = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
        if (elapsed_sec > 3000) break; 
        
        vector<double> edge_score(num_edges, 0.0);
        
        for (int sim = 0; sim < num_sims; ++sim) {
            // Hops filtering setup (only traverse valid paths given blocked edges)
            vector<int> dist_from_S(n, 1e9);
            queue<int> q;
            dist_from_S[super_source] = 0;
            q.push(super_source);
            
            while (!q.empty()) {
                int u = q.front();
                q.pop();
                if (hops != -1 && u != super_source && dist_from_S[u] >= hops) continue;
                for (const auto& edge : adj_orig[u]) {
                    if (edge.id != -1 && edge_blocked[edge.id]) continue;
                    int v = edge.dst;
                    if (dist_from_S[v] == 1e9) {
                        dist_from_S[v] = dist_from_S[u] + 1;
                        q.push(v);
                    }
                }
            }
            
            // Sample graph configuration
            // Augmented representation: nodes = original + dummy edge-nodes
            int active_edges_count = 0;
            for (int i=0; i<num_edges; i++) if(!edge_blocked[i]) active_edges_count++;
            
            int total_augmented_nodes = n + active_edges_count;
            vector<vector<int>> aug_adj(total_augmented_nodes);
            vector<vector<int>> aug_rev_adj(total_augmented_nodes);
            vector<bool> is_dummy(total_augmented_nodes, false);
            vector<int> dummy_to_edge(total_augmented_nodes, -1);
            
            int dummy_idx = n;
            
            // Re-run BFS to sample edges that are reachable
            vector<bool> visited(n, false);
            visited[super_source] = true;
            q.push(super_source);
            
            while(!q.empty()) {
                int u = q.front();
                q.pop();
                for (const auto& edge : adj_orig[u]) {
                    if (edge.id != -1 && edge_blocked[edge.id]) continue;
                    int v = edge.dst;
                    if (hops != -1 && u != super_source && dist_from_S[u] >= hops) continue;
                    
                    if (edge.p == 1.0 || dist(rng) <= edge.p) {
                        if (edge.id == -1) {
                            // Direct supersource edge without dummy
                            aug_adj[u].push_back(v);
                            aug_rev_adj[v].push_back(u);
                        } else {
                            // Augment graph
                            is_dummy[dummy_idx] = true;
                            dummy_to_edge[dummy_idx] = edge.id;
                            
                            aug_adj[u].push_back(dummy_idx);
                            aug_rev_adj[dummy_idx].push_back(u);
                            
                            aug_adj[dummy_idx].push_back(v);
                            aug_rev_adj[v].push_back(dummy_idx);
                            
                            dummy_idx++;
                        }
                        
                        if (!visited[v]) {
                            visited[v] = true;
                            q.push(v);
                        }
                    }
                }
            }
            
            // We now have augmented sampled graph, construct dominator tree
            build_dominator_tree(super_source, dummy_idx, aug_adj, aug_rev_adj);
            
            // Compute subtree sizes recursively prioritizing original node retention
            vector<int> subtree_size(dummy_idx, 0);
            if (dfn[super_source]) compute_subtree_sizes(super_source, dominator_tree, is_dummy, subtree_size);
            
            // Increment edge scores based on subtree sizes of their dummy representation
            for (int i = n; i < dummy_idx; ++i) {
                if(is_dummy[i] && dfn[i]) {
                    int eid = dummy_to_edge[i];
                    edge_score[eid] += (double)subtree_size[i] / num_sims;
                }
            }
        }
        
        // Select best edge to block
        int best_edge = -1;
        double max_sc = -1.0;
        for (int i = 0; i < num_edges; ++i) {
            if (!edge_blocked[i] && edge_score[i] > max_sc) {
                max_sc = edge_score[i];
                best_edge = i;
            }
        }
        
        if (best_edge != -1 && max_sc > 0) {
            edge_blocked[best_edge] = true;
            blocked_edges_ids.push_back(best_edge);
        } else {
            for(int i=0; i<num_edges && blocked_edges_ids.size() < k; i++) {
                if(!edge_blocked[i]) {
                    edge_blocked[i]=true;
                    blocked_edges_ids.push_back(i);
                }
            }
            break; 
        }
    }
    
    while (blocked_edges_ids.size() < k) {
        for(int i=0; i<num_edges; i++) {
            if(!edge_blocked[i]) {
                edge_blocked[i] = true;
                blocked_edges_ids.push_back(i);
                if ((int)blocked_edges_ids.size() >= k) break;
            }
        }
    }
    
    ofstream out(output_path);
    for (int i = 0; i < min((int)blocked_edges_ids.size(), k); i++) {
        int eid = blocked_edges_ids[i];
        out << raw_edges[eid].orig_u << " " << raw_edges[eid].orig_v << "\n";
    }
    out.close();
    
    return 0;
}
