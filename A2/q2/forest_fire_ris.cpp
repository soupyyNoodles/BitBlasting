#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
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

    int n = orig_id.size();
    vector<vector<Edge>> adj_rev(n);
    vector<vector<Edge>> adj_fwd(n);
    int num_edges = raw_edges.size();
    
    for (int i = 0; i < num_edges; ++i) {
        adj_rev[raw_edges[i].v].push_back({raw_edges[i].v, raw_edges[i].u, raw_edges[i].p, i});
        adj_fwd[raw_edges[i].u].push_back({raw_edges[i].u, raw_edges[i].v, raw_edges[i].p, i});
    }

    vector<bool> is_seed(n, false);
    for (int seed : initial_seeds) {
        is_seed[seed] = true;
    }

    vector<bool> edge_blocked(num_edges, false);
    vector<int> blocked_edges_ids;

    mt19937 rng(42);
    uniform_real_distribution<double> dist(0.0, 1.0);
    uniform_int_distribution<int> dist_node(0, n - 1);

    int num_rr_sets = max(n_random_instances * 10, 10000); // Need many RR sets for statistical significance

    for (int step = 0; step < k; ++step) {
        auto current_time = chrono::steady_clock::now();
        double elapsed_sec = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
        if (elapsed_sec > 3000) break;
        
        // Hops constraint mechanism
        vector<int> dist_from_S(n, 1e9);
        queue<int> q_fwd;
        for (int seed : initial_seeds) {
            dist_from_S[seed] = 0;
            q_fwd.push(seed);
        }
        while (!q_fwd.empty()) {
            int u = q_fwd.front();
            q_fwd.pop();
            if (hops != -1 && dist_from_S[u] >= hops) continue;
            for (const auto& edge : adj_fwd[u]) {
                if (edge_blocked[edge.id]) continue;
                int v = edge.dst;
                if (dist_from_S[v] == 1e9) {
                    dist_from_S[v] = dist_from_S[u] + 1;
                    q_fwd.push(v);
                }
            }
        }
        
        vector<double> edge_score(num_edges, 0.0);
        
        for (int i = 0; i < num_rr_sets; ++i) {
            int target = dist_node(rng);
            
            // If hops bounded, we only care about nodes that are reachable within hop limits anyway
            if (hops != -1 && dist_from_S[target] > hops) continue;
            
            vector<int> q;
            q.push_back(target);
            vector<bool> visited(n, false);
            visited[target] = true;
            
            vector<int> sampled_edges;
            bool reached_seed = false;
            
            int head = 0;
            while (head < (int)q.size()) {
                int u = q[head++];
                
                if (is_seed[u]) {
                    reached_seed = true;
                    // We can stop here since we just need to know if it's reachable from *any* seed
                    break;
                }
                
                for (const auto& edge : adj_rev[u]) {
                    if (edge_blocked[edge.id]) continue;
                    
                    int v = edge.dst; // in reverse graph, dst is the source
                    
                    // In reverse BFS, the node must be capable of reaching target within remaining hops. 
                    // dist_from_S guarantees it's reachable from seed within hops. 
                    if (hops != -1 && dist_from_S[v] >= dist_from_S[u]) continue; 
                    
                    if (!visited[v] && dist(rng) <= edge.p) {
                        visited[v] = true;
                        q.push_back(v);
                        sampled_edges.push_back(edge.id);
                    }
                }
            }
            
            if (reached_seed) {
                // Technically we should only count edges on the path from seed to target.
                // The RR set conceptually contains all these reverse edges.
                for (int eid : sampled_edges) {
                    edge_score[eid] += 1.0;
                }
            }
        }
        
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
