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
    vector<vector<Edge>> adj(n);
    int num_edges = raw_edges.size();
    
    for (int i = 0; i < num_edges; ++i) {
        adj[raw_edges[i].u].push_back({raw_edges[i].u, raw_edges[i].v, raw_edges[i].p, i});
    }

    vector<bool> edge_blocked(num_edges, false);
    vector<int> blocked_edges_ids;

    mt19937 rng(42);
    uniform_real_distribution<double> dist(0.0, 1.0);

    // Number of sims per selection: 
    // higher ensures better accuracy.
    int num_sims = max(n_random_instances, 200);

    for (int step = 0; step < k; ++step) {
        // precompute h-hop validity
        vector<int> dist_from_S(n, 1e9);
        queue<int> q;
        for (int seed : initial_seeds) {
            dist_from_S[seed] = 0;
            q.push(seed);
        }
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            if (hops != -1 && dist_from_S[u] >= hops) continue;
            for (const auto& edge : adj[u]) {
                if (edge_blocked[edge.id]) continue;
                int v = edge.dst;
                if (dist_from_S[v] == 1e9) {
                    dist_from_S[v] = dist_from_S[u] + 1;
                    q.push(v);
                }
            }
        }
        
        vector<double> edge_score(num_edges, 0.0);
        
        auto current_time = chrono::steady_clock::now();
        double elapsed_sec = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
        if (elapsed_sec > 3000) break; // Timeout safeguard
        
        int dyn_sims = num_sims;
        if (n_random_instances < 50) dyn_sims = 50; 
        
        // Simulating
        for (int sim = 0; sim < dyn_sims; ++sim) {
            vector<int> burn_time(n, -1);
            vector<int> b_curr;
            for (int seed : initial_seeds) {
                burn_time[seed] = 0;
                b_curr.push_back(seed);
            }
            
            vector<vector<int>> igniters(n);
            int current_t = 0;
            
            while (!b_curr.empty()) {
                vector<int> b_next;
                for (int u : b_curr) {
                    for (const auto& edge : adj[u]) {
                        if (edge_blocked[edge.id]) continue;
                        int v = edge.dst;
                        if (hops != -1 && dist_from_S[v] > hops) continue;
                        if (burn_time[v] != -1 && burn_time[v] <= current_t) continue;
                        
                        if (dist(rng) <= edge.p) {
                            igniters[v].push_back(edge.id);
                            if (burn_time[v] == -1) {
                                burn_time[v] = current_t + 1;
                                b_next.push_back(v);
                            }
                        }
                    }
                }
                b_curr = b_next;
                current_t++;
            }
            
            // Assign credits backwards
            vector<double> node_credit(n, 1.0);
            for (int seed : initial_seeds) {
                node_credit[seed] = 0.0; 
            }
            
            vector<vector<int>> nodes_at_time(current_t + 1);
            for (int i = 0; i < n; ++i) {
                if (burn_time[i] > 0) {
                    nodes_at_time[burn_time[i]].push_back(i);
                }
            }
            
            for (int t = current_t; t >= 1; --t) {
                for (int v : nodes_at_time[t]) {
                    double c = node_credit[v];
                    int ways = igniters[v].size();
                    if (ways > 0) {
                        double split = c / ways;
                        for (int eid : igniters[v]) {
                            edge_score[eid] += split;
                            int u = raw_edges[eid].u;
                            node_credit[u] += split;
                        }
                    }
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
            
            ofstream out(output_path);
            for (int eid : blocked_edges_ids) {
                out << raw_edges[eid].orig_u << " " << raw_edges[eid].orig_v << "\n";
            }
            out.close();
        } else {
            // Fill remaining if disconnected
            for(int i=0; i<num_edges && blocked_edges_ids.size() < k; i++) {
                if(!edge_blocked[i]) {
                    edge_blocked[i]=true;
                    blocked_edges_ids.push_back(i);
                }
            }
            ofstream out(output_path);
            for (int eid : blocked_edges_ids) {
                out << raw_edges[eid].orig_u << " " << raw_edges[eid].orig_v << "\n";
            }
            out.close();
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
