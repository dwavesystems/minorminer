// Copyright 2017 - 2020 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

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

#include "chain.hpp"
#include "embedding.hpp"
#include "embedding_problem.hpp"
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

// look at me, forward-declaring liek an adult
template <typename T>
class pathfinder_base;
template <typename T>
class pathfinder_serial;
template <typename T>
class pathfinder_parallel;

class pathfinder_public_interface {
  public:
    virtual int heuristicEmbedding() = 0;
    virtual const chain &get_chain(int) const = 0;
    virtual ~pathfinder_public_interface(){};
    virtual void set_initial_chains(map<int, vector<int>>) = 0;
    virtual void quickPass(const vector<int> &, int, int, bool, bool, double) = 0;
    virtual void quickPass(VARORDER, int, int, bool, bool, double) = 0;
};

template <typename embedding_problem_t>
class pathfinder_base : public pathfinder_public_interface {
    friend class pathfinder_serial<embedding_problem_t>;
    friend class pathfinder_parallel<embedding_problem_t>;

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

    vector<int> min_list;

    vector<distance_t> qubit_weight;

    vector<int> tmp_stats;
    vector<int> best_stats;

    int pushback;

    clock::time_point stoptime;

    vector<vector<int>> visited_list;

    vector<vector<distance_t>> distances;
    vector<vector<int>> qubit_permutations;

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
              min_list(num_qubits, 0),
              qubit_weight(num_qubits, 0),
              tmp_stats(),
              best_stats(),
              visited_list(num_vars + num_fixed, vector<int>(num_qubits)),
              distances(num_vars + num_fixed, vector<distance_t>(num_qubits + num_reserved, 0)),
              qubit_permutations() {
        vector<int> permutation(num_qubits);
        for (int q = num_qubits; q--;) permutation[q] = q;
        for (int v = num_vars + num_reserved; v--;) {
            ep.shuffle(permutation.begin(), permutation.end());
            qubit_permutations.push_back(permutation);
        }
    }

    //! setter for the initial_chains parameter
    virtual void set_initial_chains(map<int, vector<int>> chains) override {
        initEmbedding = embedding_t(ep, params.fixed_chains, chains);
    }

    virtual ~pathfinder_base() {}

    //! nonzero return if this is an improvement on our previous best embedding
    bool check_improvement(const embedding_t &emb) {
        bool better = 0;
        int embedded = emb.statistics(tmp_stats);
        if (embedded > ep.embedded) {
            ep.major_info("embedding found.\n");
            better = true;
            ep.embedded = 1;
        }
        if (embedded < ep.embedded) return 0;
        int minorstat = tmp_stats.back();
        int major = static_cast<int>(best_stats.size()) - static_cast<int>(tmp_stats.size());
        int minor = (best_stats.size() == 0) ? 0 : best_stats.back() - minorstat;

        better |= (major > 0) || (best_stats.size() == 0);
        if (better) {
            if (ep.embedded) {
                ep.major_info("max chain length %d; num max chains=%d\n", static_cast<int>(tmp_stats.size()) - 1,
                              minorstat);
                ep.target_chainsize = static_cast<int>(tmp_stats.size()) - 1;
            } else {
                ep.major_info("max qubit fill %d; num maxfull qubits=%d\n", static_cast<int>(tmp_stats.size()) + 1,
                              minorstat);
            }
        }
        if ((!better) && (major == 0) && (minor > 0)) {
            if (ep.embedded) {
                ep.minor_info("    num max chains=%d\n", minorstat);
            } else {
                ep.minor_info("    num max qubits=%d\n", minorstat);
            }
            better = true;
        }
        if (!better && (major == 0) && (minor == 0)) {
            for (size_t i = tmp_stats.size(); i--;) {
                if (tmp_stats[i] == best_stats[i]) continue;
                if (tmp_stats[i] < best_stats[i]) better = 1;
                break;
            }
        }
        if (better) {
            bestEmbedding = emb;
            tmp_stats.swap(best_stats);
        }
        return better;
    }

    //! chain accessor
    virtual const chain &get_chain(int u) const override { return bestEmbedding.get_chain(u); }

  protected:
    //! tear out and replace the chain in `emb` for variable `u`
    int find_chain(embedding_t &emb, const int u) {
        if (ep.embedded || ep.desperate) emb.steal_all(u);
        if (ep.embedded) {
            find_short_chain(emb, u, ep.target_chainsize);
            return 1;
        } else {
            emb.tear_out(u);
            return find_chain(emb, u, ep.target_chainsize);
        }
    }

    //! internal function to check if we're supposed to stop for an external reason -- namely
    //! if we've timed out (which we catch immediately and return -2 to allow the heuristic to
    //! terminate gracefully), or received a keyboard interrupt (which we allow to propagate
    //! back to the user).  If neither stopping condition is encountered, return `return_value`.
    inline int check_stops(const int &return_value) {
        try {
            params.localInteractionPtr->cancelled(stoptime);
        } catch (const TimeoutException & /*e*/) {
            ep.major_info("problem timed out");
            return -2;
        } catch (const ProblemCancelledException & /*e*/) {
            ep.major_info("problem cancelled via keyboard interrupt");
            if (params.interactive)
                return -2;
            else
                throw;
        }
        return return_value;
    }

    //! sweep over all variables, either keeping them if they are pre-initialized and connected,
    //! and otherwise finding new chains for them (each, in turn, seeking connection only with
    //! neighbors that already have chains)
    int initialization_pass(embedding_t &emb) {
        for (auto &u : ep.var_order((params.restrict_chains.size()) ? VARORDER_DFS : VARORDER_PFS)) {
            if (emb.chainsize(u) && emb.linked(u)) {
                ep.debug("chain for %d kept during initialization\n", u);
            } else {
                ep.debug("finding a new chain for %d\n", u);
                if (!find_chain(emb, u)) return -1;
            }
        }
        return check_stops(1);
    }

    //! tear up and replace each variable
    int improve_overfill_pass(embedding_t &emb) {
        bool improved = false;
        for (auto &u : ep.var_order(VARORDER_PFS)) {
            ep.debug("finding a new chain for %d\n", u);
            if (!find_chain(emb, u)) return -1;

            improved |= check_improvement(emb);
            if (ep.embedded) break;
        }
        return check_stops(static_cast<int>(improved));
    }

    //! tear up and replace each chain, strictly improving or maintaining the
    //! maximum qubit fill seen by each chain
    int pushdown_overfill_pass(embedding_t &emb) {
        int oldbound = ep.weight_bound;

        bool improved = false;
        for (auto &u : ep.var_order()) {
            if (pushback < num_vars) {
                ep.debug("finding a new chain for %d (pushdown)\n", u);
                int maxfill = 0;
                emb.steal_all(u);
                for (auto &q : emb.get_chain(u)) maxfill = max(maxfill, emb.weight(q));
                ep.debug("maxfill is %d\n", maxfill);
                ep.weight_bound = max(0, maxfill);
                emb.freeze_out(u);
                if (!find_chain(emb, u, 0)) {
                    ep.debug("pushdown bounced!\n", u);
                    pushback += 3;
                    emb.thaw_back(u);
                    emb.flip_back(u, 0);
                }
            } else {
                ep.debug("finding a new chain for %d (pushdown bypass)\n", u);
                ep.weight_bound = oldbound;
                emb.steal_all(u);
                emb.tear_out(u);
                if (!find_chain(emb, u, 0)) {
                    return -1;
                }
            }
            improved |= check_improvement(emb);
            if (ep.embedded) break;
        }
        ep.weight_bound = oldbound;
        if (!improved) pushback += (num_vars * 2) / params.inner_rounds;
        return check_stops(improved);
    }

    //! tear up and replace each chain, attempting to rebalance the chains and
    //! lower the maximum chainlength
    int improve_chainlength_pass(embedding_t &emb) {
        bool improved = false;
        ep.shuffle(qubit_permutations[0].begin(), qubit_permutations[0].end());
        std::fill(qubit_permutations.begin() + 1, qubit_permutations.end(), qubit_permutations[0]);
        for (auto &u : ep.var_order(ep.improved ? VARORDER_KEEP : VARORDER_PFS)) {
            ep.debug("finding a new chain for %d\n", u);
            if (!find_chain(emb, u)) return -1;

            improved |= check_improvement(emb);
        }
        return check_stops(improved);
    }

    //! incorporate the qubit weights associated with the chain for `v` into
    //! `total_distance`
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

    //! incorporate the distances associated with the chain for `v` into
    //! `total_distance`
    void accumulate_distance(const embedding_t &emb, const int v, vector<int> &visited, const int start,
                             const int stop) {
        auto dist = distances[v];
        for (int q = start; q < stop; q++) {
            if ((visited[q] == 1) && (total_distance[q] != max_distance) && !(ep.reserved(q)) &&
                (dist[q] != max_distance) && emb.weight(q) < ep.weight_bound) {
                total_distance[q] += dist[q];
            } else {
                total_distance[q] = max_distance;
            }
        }
    }

    //! a wrapper for `accumulate_distance` and `accumulate_distance_at_chain`
    inline void accumulate_distance(const embedding_t &emb, const int v, vector<int> &visited) {
        accumulate_distance_at_chain(emb, v);
        accumulate_distance(emb, v, visited, 0, num_qubits);
    }

  private:
    //! compute the distances from all neighbors of `u` to all qubits
    virtual void prepare_root_distances(const embedding_t &emb, const int u) = 0;

    //! after `u` has been torn out, perform searches from each neighboring chain,
    //! select a minimum-distance root, and construct the chain
    int find_chain(embedding_t &emb, const int u, int target_chainsize) {
        // HEURISTIC WACKINESS
        // we've already got a huge amount of entropy inside these queues,
        // so we just swap out the queues -- this costs a very few operations,
        // and the impact is that parent selection in compute_distances_from_chain
        // will be altered for at least one neighbor per pass.
        auto &nbrs = ep.var_neighbors(u, rndswap_first{});
        if (nbrs.size() > 0) {
            int v = nbrs[ep.randint(0, static_cast<int>(nbrs.size() - 1))];
            qubit_permutations[u].swap(qubit_permutations[v]);
        }

        prepare_root_distances(emb, u);

        // select a random root among those qubits at minimum heuristic distance
        collectMinima(total_distance, min_list);

        int q0 = min_list[ep.randint(0, static_cast<int>(min_list.size()) - 1)];
        if (total_distance[q0] == max_distance) return 0;  // oops all qubits were overfull or unreachable

        emb.construct_chain_steiner(u, q0, parents, distances, visited_list);
        emb.flip_back(u, target_chainsize);

        return 1;
    }

    //! after `u` has been torn out, perform searches from each neighboring chain,
    //! iterating over potential roots to find a root with a smallest-possible actual chainlength
    //! whereas other variants of `find_chain` simply pick a random root candidate with minimum
    //! estimated chainlength.  this procedure takes quite a long time and requires that `emb` is
    //! a valid embedding with no overlaps.
    void find_short_chain(embedding_t &emb, const int u, const int target_chainsize) {
        int last_size = emb.freeze_out(u);
        auto &counts = total_distance;
        counts.assign(num_qubits, 0);
        unsigned int best_size = std::numeric_limits<unsigned int>::max();
        int q, degree = static_cast<int>(ep.var_neighbors(u).size());
        distance_t d;

        unsigned int stopcheck = static_cast<unsigned int>(max(last_size, target_chainsize));

        vector<distance_queue> PQ;
        PQ.reserve(ep.var_neighbors(u).size());
        for (auto &v : ep.var_neighbors(u, shuffle_first{})) {
            PQ.emplace_back(num_qubits);
            ep.prepare_visited(visited_list[v], u, v);
            dijkstra_initialize_chain(emb, v, parents[v], visited_list[v], PQ.back(), embedded_tag{});
        }
        for (distance_t D = 0; D <= last_size; D++) {
            int v_i = 0;
            for (auto &v : ep.var_neighbors(u)) {
                auto &pq = PQ[v_i++];
                auto &parent = parents[v];
                auto &permutation = qubit_permutations[v];
                auto &distance = distances[v];
                auto &visited = visited_list[v];
                while (!pq.empty()) {
                    auto z = pq.top();
                    if (z.dist > D) break;
                    q = z.node;
                    distance[q] = d = z.dist;
                    pq.pop();
                    if (!emb.weight(q)) counts[q]++;

                    if (counts[q] == degree) {
                        emb.construct_chain_steiner(u, q, parents, distances, visited_list);
                        unsigned int cs = emb.chainsize(u);
                        if (cs < best_size) {
                            best_size = cs;
                            if (best_size < stopcheck) goto finish;
                            emb.freeze_out(u);
                        } else {
                            emb.tear_out(u);
                        }
                    }

                    d += 1;
                    visited[q] = 1;
                    for (auto &p : ep.qubit_neighbors(q)) {
                        if (!visited[p]) {
                            visited[p] = 1;
                            if (!emb.weight(p)) {
                                parent[p] = q;
                                pq.emplace(p, permutation[p], d);
                            }
                        }
                    }
                }
            }
        }
        emb.thaw_back(u);
    finish:
        emb.flip_back(u, target_chainsize);
    }

  private:
    struct embedded_tag {};
    struct default_tag {};

    //! this function prepares the parent & distance-priority-queue before running dijkstra's algorithm
    //!
    template <typename pq_t, typename behavior_tag>
    void dijkstra_initialize_chain(const embedding_t &emb, const int &v, vector<int> &parent, vector<int> &visited,
                                   pq_t &pq, behavior_tag) {
        static_assert(std::is_same<behavior_tag, embedded_tag>::value || std::is_same<behavior_tag, default_tag>::value,
                      "unknown behavior tag");
        auto &permutation = qubit_permutations[v];

        // scan through the qubits.
        // * qubits in the chain of v have distance 0,
        // * overfull qubits are tagged as visited with a special value of -1
        if (ep.fixed(v)) {
            for (auto &q : emb.get_chain(v)) {
                parent[q] = -1;
                for (auto &p : ep.qubit_neighbors(q)) {
                    if (visited[p]) continue;
                    if (std::is_same<behavior_tag, embedded_tag>::value)
                        if (emb.weight(p) == 0) {
                            pq.emplace(p, permutation[p], 1);
                            parent[p] = q;
                            visited[p] = 1;
                        }
                    if (std::is_same<behavior_tag, default_tag>::value) {
                        pq.emplace(p, permutation[p], qubit_weight[p]);
                        parent[p] = q;
                        visited[p] = 1;
                    }
                }
            }
        } else {
            for (auto &q : emb.get_chain(v)) {
                pq.emplace(q, permutation[q], 0);
                parent[q] = -1;
                visited[q] = 1;
            }
        }
    }

  protected:
    //! run dijkstra's algorithm, seeded at the chain for `v`, using the `visited` vector
    //! note: qubits are only visited if `visited[q] = 1`.  the value `-1` is used to prevent
    //! searching of overfull qubits
    void compute_distances_from_chain(const embedding_t &emb, const int &v, vector<int> &visited) {
        distance_queue pq(num_qubits);
        auto &parent = parents[v];
        auto &permutation = qubit_permutations[v];
        auto &distance = distances[v];

        dijkstra_initialize_chain(emb, v, parent, visited, pq, default_tag{});

        // this is a vanilla implementation of node-weight dijkstra -- probably where we spend the most time.
        while (!pq.empty()) {
            auto z = pq.top();
            pq.pop();
            distance[z.node] = z.dist;
            for (auto &p : ep.qubit_neighbors(z.node)) {
                if (!visited[p]) {
                    visited[p] = 1;
                    if (emb.weight(p) >= ep.weight_bound) {
                        distance[p] = max_distance;
                    } else {
                        parent[p] = z.node;
                        pq.emplace(p, permutation[p], z.dist + qubit_weight[p]);
                    }
                }
            }
        }
    }

    //! compute the weight of each qubit, first selecting `alpha`
    void compute_qubit_weights(const embedding_t &emb) {
        // first, find the maximum value of alpha that won't result in arithmetic overflow
        int maxwid = emb.max_weight();
        ep.populate_weight_table(maxwid);
        compute_qubit_weights(emb, 0, num_qubits);
    }

    //! compute the weight of each qubit in the range from `start` to `stop`,
    //! where the weight is `2^(alpha*fill)` where `fill` is the number of
    //! chains which use that qubit
    void compute_qubit_weights(const embedding_t &emb, const int start, const int stop) {
        for (int q = start; q < stop; q++) qubit_weight[q] = ep.weight(emb.weight(q));
    }

  public:
    virtual void quickPass(VARORDER varorder, int chainlength_bound, int overlap_bound, bool local_search,
                           bool clear_first, double round_beta) override {
        const vector<int> &vo = ep.var_order(varorder);
        if (vo.size() == 0)
            throw BadInitializationException(
                    "the variable ordering has length zero, did you attempt VARORDER_KEEP without running another "
                    "strategy first?");

        else
            quickPass(vo, chainlength_bound, overlap_bound, local_search, clear_first, round_beta);
    }

    virtual void quickPass(const vector<int> &varorder, int chainlength_bound, int overlap_bound, bool local_search,
                           bool clear_first, double round_beta) override {
        int lastsize, got;
        int old_bound = ep.weight_bound;
        ep.weight_bound = 1 + overlap_bound;
        ep.round_beta = round_beta;
        if (clear_first) bestEmbedding = initEmbedding;
        for (auto &u : varorder) {
            lastsize = bestEmbedding.chainsize(u);
            if (lastsize) {
                bestEmbedding.steal_all(u);
                lastsize = bestEmbedding.chainsize(u);
            }

            bool ls = local_search;
            if (ls && lastsize) {
                for (auto &q : bestEmbedding.get_chain(u)) {
                    if (bestEmbedding.weight(q) > 1) {
                        ls = false;
                        break;
                    }
                }
            }

            if (ls) {
                got = 1;
                find_short_chain(bestEmbedding, u, chainlength_bound);
            } else {
                if (lastsize) bestEmbedding.tear_out(u);
                got = find_chain(bestEmbedding, u, chainlength_bound);
            }

            if (got) {
                if (chainlength_bound > 0 &&
                    bestEmbedding.chainsize(u) > static_cast<unsigned int>(chainlength_bound)) {
                    bestEmbedding.steal_all(u);
                    bestEmbedding.tear_out(u);
                }
            }
        }
        ep.weight_bound = old_bound;
    }

    //! perform the heuristic embedding, returning 1 if an embedding was found and 0 otherwise
    virtual int heuristicEmbedding() override {
        auto timeout0 = duration<double>(params.timeout);
        auto timeout = duration_cast<clock::duration>(timeout0);
        stoptime = clock::now() + timeout;
        ep.reset_mood();
        if (params.skip_initialization) {
            if (initEmbedding.linked()) {
                currEmbedding = initEmbedding;
            } else {
                throw BadInitializationException(
                        "cannot bootstrap from initial embedding.  "
                        "disable skip_initialization or throw this embedding away");
            }
        } else {
            currEmbedding = initEmbedding;
            switch (initialization_pass(currEmbedding)) {
                case -2:
                    return 0;
                case -1:
                    throw BadInitializationException(
                            "Failed during initialization.  This typically "
                            "occurs when the source graph is unreasonably large or when the embedding "
                            "problem is over-constrained (via max_fill, initial_chains, fixed_chains, "
                            "and/or restrict_chains).");
            }
        }
        ep.major_info("initialized\n");
        ep.initialized = 1;
        best_stats.clear();
        check_improvement(currEmbedding);
        ep.improved = 1;
        currEmbedding = bestEmbedding;
        for (int trial_patience = params.tries; trial_patience-- && (!ep.embedded);) {
            int improvement_patience = params.max_no_improvement;
            ep.major_info("embedding trial %d\n", params.tries - trial_patience);
            pushback = 0;
            for (int round_patience = params.inner_rounds;
                 round_patience-- && improvement_patience && (!ep.embedded);) {
                int r;
                ep.extra_info("overfill improvement pass (%d more before giving up on this trial)\n",
                              min(improvement_patience, round_patience) - 1);
                ep.extra_info("max qubit fill %d, num max qubits %d\n", best_stats.size() + 1, best_stats.back());
                ep.desperate = (improvement_patience <= 1) | (!trial_patience) | (!round_patience);
                if (pushback < num_vars) {
                    r = pushdown_overfill_pass(currEmbedding);
                } else {
                    pushback--;
                    r = improve_overfill_pass(currEmbedding);
                }
                switch (r) {
                    case -2:
                        improvement_patience = 0;
                        break;
                    case -1:
                        currEmbedding = bestEmbedding;  // fallthrough
                    case 0:
                        improvement_patience--;
                        ep.improved = 0;
                        break;
                    case 1:
                        improvement_patience = params.max_no_improvement;
                        pushback = 0;
                        ep.improved = 1;
                        break;
                }
            }
            if (trial_patience && !ep.embedded && !improvement_patience) {
                ep.initialized = ep.desperate = pushback = 0;
                currEmbedding = initEmbedding;
                int r = initialization_pass(currEmbedding);
                switch (r) {
                    case -2:
                        trial_patience = 0;
                        break;
                    case -1:
                        currEmbedding = bestEmbedding;
                        break;
                    case 1:
                        best_stats.clear();  // overwrite bestEmbedding for a real restart
                        check_improvement(currEmbedding);
                        break;
                }
                ep.initialized = 1;
            }
        }

        if (ep.embedded && params.chainlength_patience) {
            ep.major_info("reducing chain lengths\n");
            int improvement_patience = params.chainlength_patience;
            ep.weight_bound = 1;
            currEmbedding = bestEmbedding;
            while (improvement_patience) {
                lastEmbedding = currEmbedding;
                ep.extra_info("chainlength improvement pass (%d more before giving up)\n", improvement_patience - 1);
                ep.extra_info("max chain length %d, num of max chains %d\n", best_stats.size() - 1, best_stats.back());
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
                        ep.improved = 0;
                        break;
                    case 1:
                        improvement_patience = params.chainlength_patience;
                        ep.improved = 1;
                        break;
                }
            }
        }
        return ep.embedded;
    }
};

//! A pathfinder where the Dijkstra-from-neighboring-chain passes are done serially.
template <typename embedding_problem_t>
class pathfinder_serial : public pathfinder_base<embedding_problem_t> {
  public:
    using super = pathfinder_base<embedding_problem_t>;
    using embedding_t = embedding<embedding_problem_t>;

  private:
  public:
    pathfinder_serial(optional_parameters &p_, int n_v, int n_f, int n_q, int n_r, vector<vector<int>> &v_n,
                      vector<vector<int>> &q_n)
            : super(p_, n_v, n_f, n_q, n_r, v_n, q_n) {}
    virtual ~pathfinder_serial() {}

    virtual void prepare_root_distances(const embedding_t &emb, const int u) override {
        super::ep.prepare_distances(super::total_distance, u, max_distance);
        super::compute_qubit_weights(emb);

        // run Dijkstra's algorithm from each neighbor to compute distances and shortest paths to neighbor's chains
        int neighbors_embedded = 0;
        for (auto &v : super::ep.var_neighbors(u)) {
            if (!emb.chainsize(v)) continue;
            neighbors_embedded++;
            super::ep.prepare_visited(super::visited_list[v], u, v);
            super::compute_distances_from_chain(emb, v, super::visited_list[v]);
            super::accumulate_distance(emb, v, super::visited_list[v]);
        }

        if (!neighbors_embedded) {
            for (int q = super::num_qubits; q--;)
                if (emb.weight(q) >= super::ep.weight_bound) {
                    super::total_distance[q] = max_distance;
                } else {
                    distance_t d = max(super::qubit_weight[q], super::total_distance[q]);
                    super::total_distance[q] = d;
                }
        }
    }
};

//! A pathfinder where the Dijkstra-from-neighboring-chain passes are done serially.
template <typename embedding_problem_t>
class pathfinder_parallel : public pathfinder_base<embedding_problem_t> {
  public:
    using super = pathfinder_base<embedding_problem_t>;
    using embedding_t = embedding<embedding_problem_t>;

  private:
    int num_threads;
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

            vector<int> &visited = super::visited_list[v];
            super::ep.prepare_visited(visited, u, v);
            super::compute_distances_from_chain(emb, v, visited);

            get_job.lock();
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
              futures(num_threads),
              thread_weight(num_threads) {}
    virtual ~pathfinder_parallel() {}

    virtual void prepare_root_distances(const embedding_t &emb, const int u) override {
        exec_indexed([this, &emb](int i, int a, int b) { thread_weight[i] = emb.max_weight(a, b); });

        int maxwid = *std::max_element(begin(thread_weight), end(thread_weight));
        super::ep.populate_weight_table(maxwid);

        exec_chunked([this, &emb, u](int a, int b) {
            super::compute_qubit_weights(emb, a, b);
            this->ep.prepare_distances(this->total_distance, u, max_distance, a, b);
        });

        nbr_i = 0;
        neighbors_embedded = 0;
        for (int i = 0; i < num_threads; i++)
            futures[i] = std::async(std::launch::async, [this, &emb, &u]() { run_in_thread(emb, u); });
        for (int i = 0; i < num_threads; i++) futures[i].wait();

        for (auto &v : super::ep.var_neighbors(u)) {
            super::accumulate_distance_at_chain(emb, v);  // this isn't parallel but at least it should be sparse?
        }

        exec_chunked([this, &emb, u](int a, int b) {
            for (auto &v : super::ep.var_neighbors(u)) {
                if (emb.chainsize(v)) {
                    this->accumulate_distance(emb, v, super::visited_list[v], a, b);
                }
            }
            if (!neighbors_embedded)
                for (int q = a; q < b; q++)
                    if (emb.weight(q) >= super::ep.weight_bound) super::total_distance[q] = max_distance;
        });
    }
};
}  // namespace find_embedding
