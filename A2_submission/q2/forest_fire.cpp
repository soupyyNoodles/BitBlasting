#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

#ifndef HYBRID_USE_REACHABILITY_FILTER
#define HYBRID_USE_REACHABILITY_FILTER 1
#endif

#ifndef HYBRID_USE_ADAPTIVE_BUDGET
#define HYBRID_USE_ADAPTIVE_BUDGET 0
#endif

#ifndef HYBRID_USE_FINE_RERANK
#define HYBRID_USE_FINE_RERANK 1
#endif

struct Edge {
    int src;
    int dst;
    double p;
    long long orig_u;
    long long orig_v;
};

static const double kHardTimeLimitSec = 3500.0;   // under 60-min judge limit
static const double kCheckpointSec = 30.0;        // periodic partial writes

static inline double elapsed_seconds(const chrono::steady_clock::time_point& start_time) {
    auto now = chrono::steady_clock::now();
    return chrono::duration_cast<chrono::duration<double>>(now - start_time).count();
}

static void write_partial_output(const string& output_path, const vector<int>& blocked_ids,
                                 const vector<Edge>& edges, int k_cap) {
    ofstream out(output_path);
    int lim = min((int)blocked_ids.size(), k_cap);
    for (int i = 0; i < lim; ++i) {
        int eid = blocked_ids[i];
        out << edges[eid].orig_u << " " << edges[eid].orig_v << "\n";
    }
}

static void maybe_checkpoint_write(const chrono::steady_clock::time_point& start_time,
                                   double& last_write_sec,
                                   const string& output_path,
                                   const vector<int>& blocked_ids,
                                   const vector<Edge>& edges,
                                   int k_cap) {
    double t = elapsed_seconds(start_time);
    if (t - last_write_sec >= kCheckpointSec) {
        write_partial_output(output_path, blocked_ids, edges, k_cap);
        last_write_sec = t;
    }
}

static void compute_reachability_dist(
    int n,
    const vector<vector<int>>& out_edges,
    const vector<Edge>& edges,
    const vector<char>& blocked,
    const vector<int>& seeds,
    int hops,
    vector<int>& dist_from_seed
) {
    const int INF = 1e9;
    fill(dist_from_seed.begin(), dist_from_seed.end(), INF);
    queue<int> q;
    for (int s : seeds) {
        if (dist_from_seed[s] != 0) {
            dist_from_seed[s] = 0;
            q.push(s);
        }
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        if (hops != -1 && dist_from_seed[u] >= hops) continue;

        for (int eid : out_edges[u]) {
            if (blocked[eid]) continue;
            int v = edges[eid].dst;
            if (dist_from_seed[v] == INF) {
                dist_from_seed[v] = dist_from_seed[u] + 1;
                q.push(v);
            }
        }
    }
}

struct CoarseWorkspace {
    vector<int> burn_depth;
    vector<double> node_credit;
    vector<vector<int>> igniters;
    vector<int> touched_nodes;
    vector<int> touched_igniters;
    vector<vector<int>> nodes_at_depth;
    vector<int> frontier;
    vector<int> next_frontier;

    explicit CoarseWorkspace(int n)
        : burn_depth(n, -1), node_credit(n, 0.0), igniters(n), nodes_at_depth(1) {}
};

static void accumulate_credit_scores(
    const vector<vector<int>>& out_edges,
    const vector<Edge>& edges,
    const vector<char>& blocked,
    const vector<char>& active_mask,
    const vector<int>& seeds,
    int hops,
    int num_sims,
    mt19937& rng,
    vector<double>& edge_score,
    CoarseWorkspace& ws,
    const chrono::steady_clock::time_point& start_time,
    double& last_write_sec,
    const string& output_path,
    const vector<int>& blocked_ids,
    int k_cap
) {
    uniform_real_distribution<double> dist01(0.0, 1.0);

    for (int sim = 0; sim < num_sims; ++sim) {
        maybe_checkpoint_write(start_time, last_write_sec, output_path, blocked_ids, edges, k_cap);

        if (ws.nodes_at_depth.size() > 1) {
            for (size_t d = 1; d < ws.nodes_at_depth.size(); ++d) {
                ws.nodes_at_depth[d].clear();
            }
        }

        ws.frontier.clear();
        ws.touched_nodes.clear();
        ws.touched_igniters.clear();

        for (int s : seeds) {
            if (ws.burn_depth[s] == -1) {
                ws.burn_depth[s] = 0;
                ws.touched_nodes.push_back(s);
                ws.frontier.push_back(s);
            }
        }

        int max_depth = 0;

        while (!ws.frontier.empty()) {
            ws.next_frontier.clear();

            for (int u : ws.frontier) {
                int du = ws.burn_depth[u];
                if (hops != -1 && du >= hops) continue;

                for (int eid : out_edges[u]) {
                    if (blocked[eid] || !active_mask[eid]) continue;

                    int v = edges[eid].dst;

                    // Burned earlier or same/earlier time already finalized.
                    if (ws.burn_depth[v] != -1 && ws.burn_depth[v] <= du) continue;

                    if (dist01(rng) <= edges[eid].p) {
                        if (ws.igniters[v].empty()) ws.touched_igniters.push_back(v);
                        ws.igniters[v].push_back(eid);

                        if (ws.burn_depth[v] == -1) {
                            int nd = du + 1;
                            ws.burn_depth[v] = nd;
                            ws.touched_nodes.push_back(v);
                            ws.next_frontier.push_back(v);

                            if ((int)ws.nodes_at_depth.size() <= nd) ws.nodes_at_depth.resize(nd + 1);
                            ws.nodes_at_depth[nd].push_back(v);
                            if (nd > max_depth) max_depth = nd;
                        }
                    }
                }
            }

            ws.frontier.swap(ws.next_frontier);
        }

        for (int v : ws.touched_nodes) ws.node_credit[v] = 1.0;
        for (int s : seeds) ws.node_credit[s] = 0.0;

        for (int d = max_depth; d >= 1; --d) {
            for (int v : ws.nodes_at_depth[d]) {
                double c = ws.node_credit[v];
                int ways = (int)ws.igniters[v].size();
                if (ways == 0 || c <= 0.0) continue;

                double split = c / ways;
                for (int eid : ws.igniters[v]) {
                    edge_score[eid] += split;
                    ws.node_credit[edges[eid].src] += split;
                }
            }
        }

        for (int v : ws.touched_igniters) ws.igniters[v].clear();
        for (int v : ws.touched_nodes) {
            ws.burn_depth[v] = -1;
            ws.node_credit[v] = 0.0;
        }
    }
}

struct FineWorkspace {
    vector<int> burn_depth;
    vector<int> touched_nodes;
    vector<int> frontier;
    vector<int> next_frontier;

    explicit FineWorkspace(int n) : burn_depth(n, -1) {}
};

static int deterministic_spread_with_live_world(
    const vector<vector<int>>& out_edges,
    const vector<Edge>& edges,
    const vector<char>& blocked,
    const vector<char>& live_edge,
    int extra_blocked_eid,
    const vector<int>& seeds,
    int hops,
    FineWorkspace& ws
) {
    ws.frontier.clear();
    ws.touched_nodes.clear();

    int burned = 0;
    for (int s : seeds) {
        if (ws.burn_depth[s] == -1) {
            ws.burn_depth[s] = 0;
            ws.touched_nodes.push_back(s);
            ws.frontier.push_back(s);
            ++burned;
        }
    }

    while (!ws.frontier.empty()) {
        ws.next_frontier.clear();

        for (int u : ws.frontier) {
            int du = ws.burn_depth[u];
            if (hops != -1 && du >= hops) continue;

            for (int eid : out_edges[u]) {
                if (eid == extra_blocked_eid) continue;
                if (blocked[eid] || !live_edge[eid]) continue;

                int v = edges[eid].dst;
                if (ws.burn_depth[v] != -1) continue;

                ws.burn_depth[v] = du + 1;
                ws.touched_nodes.push_back(v);
                ws.next_frontier.push_back(v);
                ++burned;
            }
        }

        ws.frontier.swap(ws.next_frontier);
    }

    for (int v : ws.touched_nodes) ws.burn_depth[v] = -1;
    return burned;
}

int main(int argc, char* argv[]) {
    auto start_time = chrono::steady_clock::now();
    double last_write_sec = -1e18;

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

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unordered_map<long long, int> id_map;
    vector<long long> orig_id;

    auto get_id = [&](long long x) -> int {
        auto it = id_map.find(x);
        if (it != id_map.end()) return it->second;
        int nid = (int)orig_id.size();
        id_map[x] = nid;
        orig_id.push_back(x);
        return nid;
    };

    vector<Edge> edges;
    {
        ifstream fg(graph_path);
        long long ou, ov;
        double p;
        while (fg >> ou >> ov >> p) {
            int u = get_id(ou);
            int v = get_id(ov);
            edges.push_back({u, v, p, ou, ov});
        }
    }

    int n = (int)orig_id.size();
    int m = (int)edges.size();

    vector<int> seeds;
    {
        ifstream fs(seed_path);
        long long s;
        vector<char> seen_seed(n, 0);
        while (fs >> s) {
            auto it = id_map.find(s);
            if (it == id_map.end()) continue;
            int sid = it->second;
            if (!seen_seed[sid]) {
                seen_seed[sid] = 1;
                seeds.push_back(sid);
            }
        }
    }

    // If seed list has no in-graph nodes, fire never spreads; choose first k edges.
    vector<vector<int>> out_edges(n);
    for (int eid = 0; eid < m; ++eid) {
        out_edges[edges[eid].src].push_back(eid);
    }

    vector<char> blocked(m, 0);
    vector<int> blocked_ids;
    blocked_ids.reserve(k);

    vector<int> dist_from_seed(n, (int)1e9);
    vector<char> active_mask(m, 0);

    vector<double> coarse_score(m, 0.0);

    mt19937 coarse_rng(42);
    CoarseWorkspace coarse_ws(n);
    FineWorkspace fine_ws(n);
    vector<char> live_edge(m, 0);

    auto pick_first_unblocked = [&]() -> int {
        for (int eid = 0; eid < m; ++eid) if (!blocked[eid]) return eid;
        return -1;
    };

    for (int step = 0; step < k; ++step) {
        double tnow = elapsed_seconds(start_time);
        if (tnow >= kHardTimeLimitSec) break;

        int remaining_steps = k - step;

        // Build active candidate mask.
        if (HYBRID_USE_REACHABILITY_FILTER) {
            compute_reachability_dist(n, out_edges, edges, blocked, seeds, hops, dist_from_seed);
            fill(active_mask.begin(), active_mask.end(), 0);
            for (int eid = 0; eid < m; ++eid) {
                if (blocked[eid]) continue;
                int u = edges[eid].src;
                if (dist_from_seed[u] >= (int)1e9) continue;
                if (hops != -1 && dist_from_seed[u] >= hops) continue;
                active_mask[eid] = 1;
            }
        } else {
            for (int eid = 0; eid < m; ++eid) active_mask[eid] = (!blocked[eid]);
        }

        int n_active = 0;
        for (int eid = 0; eid < m; ++eid) n_active += active_mask[eid] ? 1 : 0;

        if (n_active == 0) {
            int fill = pick_first_unblocked();
            if (fill == -1) break;
            blocked[fill] = 1;
            blocked_ids.push_back(fill);
            write_partial_output(output_path, blocked_ids, edges, k);
            last_write_sec = elapsed_seconds(start_time);
            continue;
        }

        fill(coarse_score.begin(), coarse_score.end(), 0.0);

        int coarse_sims = max(40, n_random_instances * 3);
        if (HYBRID_USE_ADAPTIVE_BUDGET) {
            double remaining_time = max(0.0, kHardTimeLimitSec - tnow);
            double time_per_step = remaining_time / max(1, remaining_steps);

            if (time_per_step < 6.0) coarse_sims = max(15, n_random_instances);
            else if (time_per_step < 15.0) coarse_sims = max(25, n_random_instances * 2);
            else coarse_sims = max(40, n_random_instances * 3);

            if (m > 400000) coarse_sims = min(coarse_sims, max(20, n_random_instances * 2));
        }

        accumulate_credit_scores(
            out_edges, edges, blocked, active_mask, seeds, hops, coarse_sims,
            coarse_rng, coarse_score, coarse_ws,
            start_time, last_write_sec, output_path, blocked_ids, k
        );

        vector<int> active_edges;
        active_edges.reserve(n_active);
        for (int eid = 0; eid < m; ++eid) if (active_mask[eid]) active_edges.push_back(eid);

        int shortlist_size = min((int)active_edges.size(), 24);
        if (HYBRID_USE_ADAPTIVE_BUDGET) {
            if (m > 400000) shortlist_size = min(shortlist_size, 12);
            if (elapsed_seconds(start_time) > 0.8 * kHardTimeLimitSec) shortlist_size = min(shortlist_size, 8);
        }
        shortlist_size = max(1, shortlist_size);

        nth_element(active_edges.begin(), active_edges.begin() + shortlist_size - 1, active_edges.end(),
                    [&](int a, int b) {
                        return coarse_score[a] > coarse_score[b];
                    });
        active_edges.resize(shortlist_size);

        int best_edge = active_edges[0];
        double best_score = coarse_score[best_edge];
        for (int eid : active_edges) {
            if (coarse_score[eid] > best_score) {
                best_score = coarse_score[eid];
                best_edge = eid;
            }
        }

        if (HYBRID_USE_FINE_RERANK && shortlist_size > 1) {
            int fine_worlds = max(6, n_random_instances / 4);
            if (HYBRID_USE_ADAPTIVE_BUDGET) {
                double remaining_time = max(0.0, kHardTimeLimitSec - elapsed_seconds(start_time));
                double time_per_step = remaining_time / max(1, remaining_steps);
                if (time_per_step < 6.0) fine_worlds = 2;
                else if (time_per_step < 12.0) fine_worlds = 4;
                else if (time_per_step < 20.0) fine_worlds = 6;
                else fine_worlds = max(8, n_random_instances / 3);
                if (m > 400000) fine_worlds = min(fine_worlds, 5);
            }

            vector<double> avg_spread(shortlist_size, 0.0);

            for (int w = 0; w < fine_worlds; ++w) {
                maybe_checkpoint_write(start_time, last_write_sec, output_path, blocked_ids, edges, k);

                uint32_t world_seed = (uint32_t)(1337u + 104729u * (uint32_t)step + 2654435761u * (uint32_t)w);
                mt19937 world_rng(world_seed);
                uniform_real_distribution<double> dist01(0.0, 1.0);

                for (int eid = 0; eid < m; ++eid) {
                    if (blocked[eid]) {
                        live_edge[eid] = 0;
                    } else {
                        live_edge[eid] = (dist01(world_rng) <= edges[eid].p) ? 1 : 0;
                    }
                }

                for (int i = 0; i < shortlist_size; ++i) {
                    int eid = active_edges[i];
                    int burned = deterministic_spread_with_live_world(
                        out_edges, edges, blocked, live_edge, eid, seeds, hops, fine_ws);
                    avg_spread[i] += burned;
                }
            }

            int rerank_edge = best_edge;
            double rerank_best = 1e100;
            for (int i = 0; i < shortlist_size; ++i) {
                double mean_spread = avg_spread[i] / max(1, fine_worlds);
                if (mean_spread < rerank_best) {
                    rerank_best = mean_spread;
                    rerank_edge = active_edges[i];
                }
            }
            best_edge = rerank_edge;
        }

        if (best_edge == -1 || blocked[best_edge]) {
            best_edge = pick_first_unblocked();
            if (best_edge == -1) break;
        }

        blocked[best_edge] = 1;
        blocked_ids.push_back(best_edge);

        write_partial_output(output_path, blocked_ids, edges, k);
        last_write_sec = elapsed_seconds(start_time);
    }

    while ((int)blocked_ids.size() < k) {
        int eid = pick_first_unblocked();
        if (eid == -1) break;
        blocked[eid] = 1;
        blocked_ids.push_back(eid);
        write_partial_output(output_path, blocked_ids, edges, k);
    }

    // Final write with exactly min(k, blocked_ids.size()) lines.
    write_partial_output(output_path, blocked_ids, edges, k);
    return 0;
}
