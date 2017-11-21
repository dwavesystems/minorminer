#pragma once

#include <algorithm>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "embedding.hpp"
#include "embedding_problem.hpp"
#include "pairing_queue.hpp"
#include "util.hpp"

namespace find_embedding {

// The pathfinder_base object manages the heuristic embedding process.  It contains various heuristic improvement
// strategies, shared data structures used for those strategies, and a master heuristic that dispatches them.
// The most basic operation is to tear out and replace a single chain.  Chain placement for a variable u is accomplished
// by initiating Dijkstra's algorithm at each embedded neighbor's chain, to compute the distance from each qubit to the
// chains in question.  Then, the sum of these distances is used as a fitness metric, to select a root qubit q for u.
// Shortest paths from each neighboring chain to u, found in Dijkstra's algorithm, are taken together to form the new
// chain for u.
// The execution of Dijkstra's algorithm is responsible for 99% of our runtime.  It's easily parallelized when variables
// have a large number of neighbors.  The serial/parallel versions occur below.

template <typename embedding_problem_t>
class pathfinder_base {
  public:
    using embedding_t = embedding<embedding_problem_t>;

  protected:
    embedding_problem_t ep;

    optional_parameters &params;

    embedding_t bestEmbedding;
    embedding_t lastEmbedding;
    embedding_t currEmbedding;
    embedding_t initEmbedding;

    int num_qubits, num_reserved;
    int num_vars, num_fixed;

    vector<vector<int>> parents;
    vector<distance_t> total_distance;

    vector<int> tmp_space;
    vector<int> min_list;

    int_queue intqueue;

    vector<distance_t> qubit_weight;

    int embeddingSum, maxBagWidth, numMaxBags, maxChainSize, numMaxChains;
    int bestEmbeddingSum, bestWidth, bestNumMaxBags, bestChainSize, bestNumMaxChains;

    int pushback;

    clock::time_point stoptime;

    vector<distance_queue> dijkstras;

  public:
    pathfinder_base(optional_parameters &p_, int &n_v, int &n_f, int &n_q, int &n_r, vector<vector<int>> &v_n,
                    vector<vector<int>> &q_n)
            : ep(p_, n_v, n_f, n_q, n_r, v_n, q_n),
              params(p_),
              bestEmbedding(ep),
              lastEmbedding(ep),
              currEmbedding(ep),
              initEmbedding(ep, params.fixed_chains, params.initial_chains),
              num_qubits(ep.num_qubits()),
              num_reserved(ep.num_reserved()),
              num_vars(ep.num_vars()),
              num_fixed(ep.num_fixed()),
              parents(num_vars + num_fixed, vector<int>(num_qubits + num_reserved, 0)),
              total_distance(num_qubits, 0),
              tmp_space(num_qubits + num_reserved, 0),
              min_list(num_qubits, 0),
              intqueue(num_qubits),
              qubit_weight(num_qubits, 0),
              dijkstras(num_vars + num_fixed, distance_queue(num_qubits)) {}

    void display_statistics(const char *stage) const {
        display_statistics(stage, bestEmbeddingSum, bestWidth, bestNumMaxBags, bestChainSize, bestNumMaxChains);
    }

    void display_statistics(const char *stage, int eSum, int bWidth, int nMaxB, int cSize, int nMaxC) const {
        if (bWidth > 1) {
            char stats[] =
                    "%s qubit fill total = %d, max overfill = %d, num maxfill = %d, max chainlength = %d, num max "
                    "chains = %d\n";
            ep.display_message(stats, stage, eSum, bWidth, nMaxB, cSize, nMaxC);
        } else {
            char stats[] = "%s qubit total = %d, max chainlength = %d, num max chains = %d\n";
            ep.display_message(stats, stage, eSum, cSize, nMaxC);
        }
    }

    const chain &get_chain(int u) const { return bestEmbedding.get_chain(u); }

  protected:
    int find_chain(embedding_t &emb, const int u) {
        if (ep.initialized) emb.covfefe(u);
        emb.tear_out(u);
        if (ep.embedded && ep.desperate)
            return find_short_chain(emb, u, 10);
        else if (ep.desperate)
            return find_short_chain(emb, u, num_qubits);
        else
            return find_chain(emb, u, ep.target_chainsize);
    }

    int initialization_pass(embedding_t &emb) {
        for (auto &u : ep.var_order(VARORDER_PFS)) {
            //                if( emb.chainsize(u) && emb.linked(u) )
            //                    continue;
            if (!find_chain(emb, u)) return -1;
        }
        if (params.localInteractionPtr->cancelled(stoptime))
            return -2;
        else
            return 1;
    }

    int improve_overfill_pass(embedding_t &emb) {
        emb.statistics(embeddingSum, maxBagWidth, numMaxBags, maxChainSize, numMaxChains);

        bool improved = false;
        for (auto &u : ep.var_order(VARORDER_SHUFFLE)) {
            if (!find_chain(emb, u)) return -1;

            emb.statistics(embeddingSum, maxBagWidth, numMaxBags, maxChainSize, numMaxChains);

            if (maxBagWidth < bestWidth || (maxBagWidth <= bestWidth && numMaxBags < bestNumMaxBags) ||
                (maxBagWidth <= bestWidth && numMaxBags <= bestNumMaxBags && maxChainSize < bestChainSize)) {
                improved = true;
                bestEmbeddingSum = embeddingSum;
                bestWidth = maxBagWidth;
                bestNumMaxBags = numMaxBags;
                bestChainSize = maxChainSize;
                bestNumMaxChains = numMaxChains;
                bestEmbedding = emb;
                if (maxBagWidth == 1) {
                    ep.display_message("embedding found\n");
                    break;
                }
            }
            if (maxBagWidth > bestWidth || numMaxBags > bestNumMaxBags + 4) {
                emb = bestEmbedding;
            }
        }
        if (params.localInteractionPtr->cancelled(stoptime))
            return -2;
        else {
            display_statistics("overfill pass");
            return improved;
        }
    }

    int pushdown_overfill_pass(embedding_t &emb) {
        emb.statistics(embeddingSum, maxBagWidth, numMaxBags, maxChainSize, numMaxChains);
        int oldbound = ep.weight_bound;

        bool improved = false;
        for (auto &u : ep.var_order(VARORDER_BFS)) {
            int r = 0;
            if (pushback < num_vars) {
                int maxfill = 0;
                emb.covfefe(u);
                for (auto &q : emb.get_chain(u)) maxfill = max(maxfill, emb.weight(q));
                if (pushback < num_vars / 2) {
                    ep.weight_bound = maxfill;
                    ep.target_chainsize = maxChainSize - 1;
                } else {
                    ep.weight_bound = maxfill + 1;
                    ep.target_chainsize = 0;
                }
                r = find_chain(emb, u);
                if (!r) pushback++;
            }
            if (!r) {
                ep.target_chainsize = 0;
                ep.weight_bound = oldbound;
                if (!find_chain(emb, u)) return -1;
            }
            emb.statistics(embeddingSum, maxBagWidth, numMaxBags, maxChainSize, numMaxChains);

            if (maxBagWidth < bestWidth || (maxBagWidth <= bestWidth && numMaxBags < bestNumMaxBags) ||
                (maxBagWidth <= bestWidth && numMaxBags <= bestNumMaxBags && maxChainSize < bestChainSize)) {
                improved = true;
                bestEmbeddingSum = embeddingSum;
                bestWidth = maxBagWidth;
                bestNumMaxBags = numMaxBags;
                bestChainSize = maxChainSize;
                bestNumMaxChains = numMaxChains;
                bestEmbedding = emb;
                if (maxBagWidth == 1) {
                    ep.display_message("embedding found\n");
                    break;
                }
            }
            if (maxBagWidth > bestWidth || numMaxBags > bestNumMaxBags + 4) {
                emb = bestEmbedding;
            }
        }
        ep.weight_bound = oldbound;
        ep.target_chainsize = 0;
        if (params.localInteractionPtr->cancelled(stoptime))
            return -2;
        else {
            display_statistics("overfill pass");
            return improved;
        }
    }

    int improve_chainlength_pass(embedding_t &emb) {
        emb.statistics(embeddingSum, maxBagWidth, numMaxBags, maxChainSize, numMaxChains);
        bool improved = false;
        for (auto &u : ep.var_order(VARORDER_SHUFFLE)) {
            ep.target_chainsize = maxChainSize - 1;

            if (!find_chain(emb, u)) return -1;

            emb.statistics(embeddingSum, maxBagWidth, numMaxBags, maxChainSize, numMaxChains);

            if (maxChainSize < bestChainSize || (maxChainSize == bestChainSize && numMaxChains < bestNumMaxChains) ||
                (maxChainSize == bestChainSize && numMaxChains == bestNumMaxChains &&
                 embeddingSum < bestEmbeddingSum)) {
                improved = true;
                bestChainSize = maxChainSize;
                bestNumMaxChains = numMaxChains;
                bestWidth = maxBagWidth;
                bestEmbeddingSum = embeddingSum;
                bestEmbedding = emb;
            }
        }
        if (params.localInteractionPtr->cancelled(stoptime))
            return -2;
        else {
            display_statistics("chainlength pass");
            return improved;
        }
    }

    void accumulate_distance_at_chain(const embedding_t &emb, const int v) {
        if (!ep.fixed(v)) {
            for (auto &q : emb.get_chain(v)) {
                auto w = qubit_weight[q];
                if ((total_distance[q] != max_distance) && !(ep.reserved(q)) && (w != max_distance) &&
                    emb.weight(q) < ep.weight_bound && w > 0)
                    total_distance[q] += w;
                else
                    total_distance[q] = max_distance;
            }
        }
    }

    void accumulate_distance(const embedding_t &emb, const int v, vector<int> &visited, const int start,
                             const int stop) {
        auto &distqueue = dijkstras[v];
        for (int q = start; q < stop; q++) {
            auto dist = distqueue.get_value(q);
            if ((visited[q] == 1) && (total_distance[q] != max_distance) && !(ep.reserved(q)) &&
                (dist != max_distance) && emb.weight(q) < ep.weight_bound && dist > 0) {
                total_distance[q] += dist;
            } else {
                total_distance[q] = max_distance;
            }
        }
    }

    inline void accumulate_distance(const embedding_t &emb, const int v, vector<int> &visited) {
        accumulate_distance_at_chain(emb, v);
        accumulate_distance(emb, v, visited, 0, num_qubits);
    }

  private:
    virtual void prepare_root_distances(const embedding_t &emb, const int u) = 0;

    int find_chain(embedding_t &emb, const int u, int target_chainsize) {
        prepare_root_distances(emb, u);

        // select a random root among those qubits at minimum heuristic distance
        collectMinima(total_distance, min_list);

        int q0 = min_list[ep.randint(min_list.size())];
        if (total_distance[q0] == max_distance) return 0;  // oops all qubits were overfull or unreachable

        emb.construct_chain(u, q0, target_chainsize, parents);

        return 1;
    }

    int find_short_chain(embedding_t &emb, const int u, int roots_to_try = 10) {
        prepare_root_distances(emb, u);

        int better = 1;
        int q0 = -1;
        int q0_size = num_qubits + num_reserved + 1;
        int tried = 0;

        while (better && tried < roots_to_try) {
            better = 0;
            collectMinima(total_distance, min_list);
            if (!min_list.size() || total_distance[min_list[0]] == max_distance) break;
            ep.shuffle(begin(min_list), end(min_list));
            for (auto &q1 : min_list) {
                if (tried++) emb.tear_out(u);

                total_distance[q1] = max_distance;
                emb.construct_chain(u, q1, num_qubits + num_reserved + 1, parents);

                if (emb.chainsize(u) < q0_size) {
                    q0_size = emb.chainsize(u);
                    q0 = q1;
                    better = 1;
                }
                if (tried >= roots_to_try) break;
            }
        }

        if (q0 == -1) return 0;
        if (tried > 1) {
            emb.tear_out(u);
            emb.construct_chain(u, q0, ep.target_chainsize, parents);
        }
        return 1;
    }

  protected:
    void compute_distances_from_chain(const embedding_t &emb, const int &v, vector<int> &visited) {
        auto &pq = dijkstras[v];
        auto &parent = parents[v];
        int q;
        distance_t d;

        pq.reset();
        // scan through the qubits.
        // * qubits in the chain of v have distance 0,
        // * overfull qubits are tagged as visited with a special value of 2
        if (ep.fixed(v)) {
            for (auto &q : emb.get_chain(v)) {
                parent[q] = -1;
                for (auto &p : ep.qubit_neighbors(q)) {
                    pq.set_value(p, qubit_weight[p]);
                    parent[p] = q;
                }
            }
        } else {
            for (auto &q : emb.get_chain(v)) {
                pq.set_value(q, 0);
                parent[q] = -1;
            }
        }

        for (q = num_qubits; q--;)
            if (emb.weight(q) >= ep.weight_bound) visited[q] = -1;

        // this is a vanilla implementation of node-weight dijkstra -- probably where we spend the most time.
        while (pq.pop_min(q, d)) {
            visited[q] = 1;
            for (auto &p : ep.qubit_neighbors(q))
                if (!visited[p])
                    if (pq.check_decrease_value(p, d + qubit_weight[p])) parent[p] = q;
        }
    }

    void compute_qubit_weights(const embedding_t &emb) {
        // first, find the maximum value of alpha that won't result in arithmetic overflow
        int maxwid = emb.max_weight();
        if (maxwid > ep.weight_bound) maxwid = ep.weight_bound - 1;
        int alpha = maxwid > 1 ? ep.alpha / maxwid : ep.alpha - 1;
        compute_qubit_weights(emb, alpha, 0, num_qubits);
    }

    void compute_qubit_weights(const embedding_t &emb, const int alpha, const int start, const int stop) {
        for (int q = start; q < stop; q++)
            qubit_weight[q] = static_cast<const distance_t>(1) << (alpha * emb.weight(q));
    }

  public:
    int heuristicEmbedding() {
        auto timeout0 = duration<double>(params.timeout);
        auto timeout = duration_cast<clock::duration>(timeout0);
        stoptime = clock::now() + timeout;

        if (params.skip_initialization) {
            if (initEmbedding.linked()) {
                bestEmbedding = initEmbedding;
            } else {
                ep.display_message(
                        "cannot bootstrap from initial embedding.  stopping.  disable skip_initialization or throw "
                        "this embedding away");
                return 0;
            }
        } else {
            bestEmbedding = initEmbedding;
            if (initialization_pass(bestEmbedding) <= 0) {
                ep.display_message("failed during initialization. embeddings may be invalid.\n");
                return 0;
            }
        }
        ep.initialized = 1;
        bestEmbedding.statistics(bestEmbeddingSum, bestWidth, bestNumMaxBags, bestChainSize, bestNumMaxChains);

        for (int trial_patience = params.tries; trial_patience-- && (bestWidth > 1);) {
            int improvement_patience = params.max_no_improvement;
            currEmbedding = lastEmbedding = bestEmbedding;
            ep.display_message("try %d\n", params.tries - trial_patience);
            pushback = 0;
            for (int round_patience = params.inner_rounds;
                 round_patience-- && improvement_patience && (bestWidth > 1);) {
                int r;
                ep.desperate = (improvement_patience <= 1) | (!trial_patience) | (!round_patience);
                if (pushback < num_vars) {
                    r = pushdown_overfill_pass(currEmbedding);
                } else {
                    pushback--;
                    r = improve_overfill_pass(currEmbedding);
                }
                switch (r) {
                    case -1:
                        ep.display_message(
                                "connectivity failure in singleVertexAdditionHeuristic.  embeddings are invalid\n");
                    case -2:
                        improvement_patience = 0;
                        break;
                    case 0:
                        improvement_patience--;
                        break;
                    case 1:
                        improvement_patience = params.max_no_improvement;
                        pushback = 0;
                        break;
                }
            }
            if (trial_patience && (bestWidth > 1) && (improvement_patience == 0)) {
                ep.initialized = 0;
                ep.desperate = 1;
                if (initialization_pass(bestEmbedding) <= 0) {
                    ep.display_message("failed during restart. embeddings may be invalid.\n");
                    return 0;
                }
                bestEmbedding.statistics(bestEmbeddingSum, bestWidth, bestNumMaxBags, bestChainSize, bestNumMaxChains);
                ep.initialized = 1;
                ep.desperate = 0;
            }
        }
        if (bestWidth == 1) ep.embedded = 1;
        if (bestWidth == 1 && !params.fast_embedding) {
            int improvement_patience = params.chainlength_patience;
            ep.weight_bound = 1;
            currEmbedding = bestEmbedding;
            while (improvement_patience) {
                lastEmbedding = currEmbedding;
                ep.desperate = (improvement_patience == 1);
                int r = improve_chainlength_pass(currEmbedding);
                switch (r) {
                    case -1:
                        currEmbedding = lastEmbedding;
                        improvement_patience--;
                        break;
                    case -2:
                        improvement_patience = 0;
                        break;  // interrupting here is inconsequential; bestEmbedding is valid
                    case 0:
                        improvement_patience--;
                        break;
                    case 1:
                        improvement_patience = params.chainlength_patience;
                        break;
                }
            }
        }
        return bestWidth == 1;
    }
};

// A pathfinder where the Dijkstra-from-neighboring-chain passes are done serially.
template <typename embedding_problem_t>
class pathfinder_serial : public pathfinder_base<embedding_problem_t> {
  public:
    using super = pathfinder_base<embedding_problem_t>;
    using embedding_t = embedding<embedding_problem_t>;

  private:
    vector<int> visited;

  public:
    pathfinder_serial(optional_parameters &p_, int n_v, int n_f, int n_q, int n_r, vector<vector<int>> &v_n,
                      vector<vector<int>> &q_n)
            : super(p_, n_v, n_f, n_q, n_r, v_n, q_n), visited(super::num_qubits + super::num_reserved) {}

    virtual void prepare_root_distances(const embedding_t &emb, const int u) override {
        super::ep.prepare_distances(super::total_distance, u, max_distance);
        super::compute_qubit_weights(emb);

        // run Dijkstra's algorithm from each neighbor to compute distances and shortest paths to neighbor's chains
        int neighbors_embedded = 0;
        for (auto &v : super::ep.var_neighbors(u)) {
            if (!emb.chainsize(v)) continue;
            neighbors_embedded++;
            super::ep.prepare_visited(visited, u, v);
            super::compute_distances_from_chain(emb, v, visited);
            super::accumulate_distance(emb, v, visited);
        }

        for (int q = super::num_qubits; q--;)
            if (emb.weight(q) >= super::ep.weight_bound) super::total_distance[q] = max_distance;
    }
};

// A pathfinder where the Dijkstra-from-neighboring-chain passes are done serially.
template <typename embedding_problem_t>
class pathfinder_parallel : public pathfinder_base<embedding_problem_t> {
  public:
    using super = pathfinder_base<embedding_problem_t>;
    using embedding_t = embedding<embedding_problem_t>;

  private:
    int num_threads;
    vector<vector<int>> visited_list;
    vector<std::future<void>> futures;
    vector<int> thread_weight;
    mutex get_job;

    unsigned int nbr_i;
    int neighbors_embedded;

    void run_in_thread(const embedding_t &emb, const int u) {
        get_job.lock();
        while (1) {
            int v = -1;
            const vector<int> &neighbors = super::ep.var_neighbors(u);
            while (nbr_i < neighbors.size()) {
                int v0 = neighbors[nbr_i++];
                if (emb.chainsize(v0)) {
                    v = v0;
                    neighbors_embedded++;
                    break;
                }
            }
            get_job.unlock();

            if (v < 0) break;

            vector<int> &visited = visited_list[v];
            super::ep.prepare_visited(visited, u, v);
            super::compute_distances_from_chain(emb, v, visited);

            get_job.lock();
            super::accumulate_distance_at_chain(emb, v);
        }
    }

    template <typename C>
    void exec_chunked(C e_chunk) {
        const int grainsize = super::num_qubits / num_threads;
        int grainmod = super::num_qubits % num_threads;

        int a = 0;
        for (int i = num_threads; i--;) {
            int b = a + grainsize + (grainmod-- > 0);
            futures[i] = std::async(std::launch::async, e_chunk, a, b);
            a = b;
        }
        for (int i = num_threads; i--;) futures[i].wait();
    }

    template <typename C>
    void exec_indexed(C e_chunk) {
        const int grainsize = super::num_qubits / num_threads;
        int grainmod = super::num_qubits % num_threads;

        int a = 0;
        for (int i = num_threads; i--;) {
            int b = a + grainsize + (grainmod-- > 0);
            futures[i] = std::async(std::launch::async, e_chunk, i, a, b);
            a = b;
        }
        for (int i = num_threads; i--;) futures[i].wait();
    }

  public:
    pathfinder_parallel(optional_parameters &p_, int n_v, int n_f, int n_q, int n_r, vector<vector<int>> &v_n,
                        vector<vector<int>> &q_n)
            : super(p_, n_v, n_f, n_q, n_r, v_n, q_n),
              num_threads(min(p_.threads, n_q)),
              visited_list(n_v + n_f, vector<int>(super::num_qubits)),
              futures(num_threads),
              thread_weight(num_threads) {}

    virtual void prepare_root_distances(const embedding_t &emb, const int u) override {
        exec_indexed([this, &emb](int i, int a, int b) { thread_weight[i] = emb.max_weight(a, b); });

        int maxwid = *std::max_element(begin(thread_weight), end(thread_weight));
        if (maxwid > super::ep.weight_bound) maxwid = super::ep.weight_bound - 1;
        int alpha = maxwid > 1 ? super::ep.alpha / maxwid : super::ep.alpha - 1;

        exec_chunked([this, &emb, alpha](int a, int b) { super::compute_qubit_weights(emb, alpha, a, b); });

        exec_chunked(
                [this, u](int a, int b) { super::ep.prepare_distances(super::total_distance, u, max_distance, a, b); });

        nbr_i = 0;
        neighbors_embedded = 0;
        for (int i = 0; i < num_threads; i++)
            futures[i] = std::async(std::launch::async, [this, &emb, &u]() { run_in_thread(emb, u); });
        for (int i = 0; i < num_threads; i++) futures[i].wait();

        for (auto &v : super::ep.var_neighbors(u)) {
            if (emb.chainsize(v)) {
                exec_chunked(
                        [this, &emb, v](int a, int b) { super::accumulate_distance(emb, v, visited_list[v], a, b); });
            }
        }
    }
};
}
