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
#include <chrono>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "util.hpp"

namespace find_embedding {

enum VARORDER { VARORDER_SHUFFLE, VARORDER_DFS, VARORDER_BFS, VARORDER_PFS, VARORDER_RPFS, VARORDER_KEEP };

// This file contains component classes for constructing embedding problems.  Presently, an embedding_problem class is
// constructed by combining the embedding_problem_base class with a domain_handler class and a fixed_handler class.
// This is used to accomplish dynamic dispatch for code in/around the inner loops without fouling their performance.

// Domain handlers are used to control which qubits are available for use for which variables.
// they perform three operations:
//   * marking unavailable qubits as "visited" so that we don't visit them in Dijkstra's algorithm
//   * setting unavailable qubits to an effectively infinite cost for root selection
//   * checking if a particular qubit is available for a particular variable

//! this is the trivial domain handler, where every variable is allowed to use every qubit
class domain_handler_universe {
  public:
    domain_handler_universe(optional_parameters & /*p*/, int /*n_v*/, int /*n_f*/, int /*n_q*/, int /*n_r*/) {}
    virtual ~domain_handler_universe() {}

    static inline void prepare_visited(vector<int> &visited, int /*u*/, int /*v*/) {
        std::fill(std::begin(visited), std::end(visited), 0);
    }

    static inline void prepare_distances(vector<distance_t> &distance, const int /*u*/, const distance_t & /*mask_d*/) {
        std::fill(std::begin(distance), std::end(distance), 0);
    }

    static inline void prepare_distances(vector<distance_t> &distance, const int /*u*/, const distance_t & /*mask_d*/,
                                         const int start, const int stop) {
        std::fill(std::begin(distance) + start, std::begin(distance) + stop, 0);
    }

    static inline bool accepts_qubit(int /*u*/, int /*q*/) { return 1; }
};

//! this domain handler stores masks for each variable so that prepare_visited and prepare_distances are barely more
//! expensive than a memcopy
class domain_handler_masked {
    optional_parameters &params;
    vector<vector<int>> masks;

  public:
    domain_handler_masked(optional_parameters &p, int n_v, int n_f, int n_q, int n_r)
            : params(p), masks(n_v + n_f, vector<int>()) {
#ifdef CPPDEBUG
        for (auto &vC : params.restrict_chains)
            for (auto &q : vC.second) minorminer_assert(0 <= q && q < n_q + n_r);
#endif
        auto nostrix = std::end(params.restrict_chains);
        for (int v = n_v + n_f; v--;) {
            auto chain = params.restrict_chains.find(v);
            if (chain != nostrix) {
                masks[v].resize(n_q + n_r, -1);
                auto vC = *chain;
                for (auto &q : vC.second) masks[v][q] = 0;
            } else {
                masks[v].resize(n_q + n_r, 0);
            }
        }
    }
    virtual ~domain_handler_masked() {}

    inline void prepare_visited(vector<int> &visited, const int u, const int v) {
        vector<int> &uMask = masks[u];
        vector<int> &vMask = masks[v];
        int *umask = uMask.data();
        int *vmask = vMask.data();
        int *vis = visited.data();
        for (int *stop = vis + visited.size(); vis < stop; ++vis, ++umask, ++vmask) *vis = (*umask) & (*vmask);
    }

    inline void prepare_distances(vector<distance_t> &distance, const int u, const distance_t &mask_d) {
        vector<int> &uMask = masks[u];
        int *umask = uMask.data();
        distance_t *dist = distance.data();
        distance_t *dend = dist + distance.size();
        for (; dist < dend; dist++, umask++) *dist = (-(*umask)) * mask_d;
    }

    inline void prepare_distances(vector<distance_t> &distance, const int u, const distance_t &mask_d, const int start,
                                  const int stop) {
        vector<int> &uMask = masks[u];
        int *umask = uMask.data() + start;
        distance_t *dist = distance.data() + start;
        distance_t *dend = distance.data() + stop;
        for (; dist < dend; dist++, umask++) *dist = (-(*umask)) * mask_d;
    }

    inline bool accepts_qubit(const int u, const int q) { return !(masks[u][q]); }
};

// Fixed handlers are used to control which variables are allowed to be torn up and replaced.  Currently, there is no
// option to fix/unfix variables after an embedding problem has been instantiated, but TODO that will be implemented in
// the future in fixed_handler_list.  Fixed variables are assumed to have chains; reserved qubits are not available for
// use in producing new chains (currently TODO this only happens because they belong to the chains of fixed variables).

//! This fixed handler is used when there are no fixed variables.
class fixed_handler_none {
  public:
    fixed_handler_none(optional_parameters & /*p*/, int /*n_v*/, int /*n_f*/, int /*n_q*/, int /*n_r*/) {}
    virtual ~fixed_handler_none() {}

    static inline bool fixed(int /*u*/) { return false; }

    static inline bool reserved(int /*u*/) { return false; }
};

//! This fixed handler is used when the fixed variables are processed before instantiation and relabeled such that
//! variables v >= num_v are fixed and qubits q >= num_q are reserved
class fixed_handler_hival {
  private:
    int num_v, num_q;

  public:
    fixed_handler_hival(optional_parameters & /*p*/, int n_v, int /*n_f*/, int n_q, int /*n_r*/)
            : num_v(n_v), num_q(n_q) {}
    virtual ~fixed_handler_hival() {}

    inline bool fixed(const int u) { return u >= num_v; }

    inline bool reserved(const int q) { return q >= num_q; }
};

//! Output handlers are used to control output.  We provide two handlers -- one which only reports all errors (and
//! optimizes away all other output) and another which provides full output.  When verbose is zero, we recommend
//! the errors-only handler and otherwise, the full handler

//! Here's the full output handler
template <bool verbose>
class output_handler {
    optional_parameters &params;

  public:
    output_handler(optional_parameters &p) : params(p) {}

    //! printf regardless of the verbosity level
    template <typename... Args>
    void error(const char *format, Args... args) const {
        params.error(format, args...);
    }

    //! printf at the major_info verbosity level
    template <typename... Args>
    void major_info(const char *format, Args... args) const {
        if (verbose && params.verbose > 0) params.major_info(format, args...);
    }

    //! print at the minor_info verbosity level
    template <typename... Args>
    void minor_info(const char *format, Args... args) const {
        if (verbose && params.verbose > 1) params.minor_info(format, args...);
    }

    //! print at the extra_info verbosity level
    template <typename... Args>
    void extra_info(const char *format, Args... args) const {
        if (verbose && params.verbose > 2) params.extra_info(format, args...);
    }

    //! print at the debug verbosity level (only works when `CPPDEBUG` is set)
    template <typename... Args>
#ifdef CPPDEBUG
    void debug(const char *format, Args... args) const {
        if (verbose && params.verbose > 3) {
            params.debug(format, args...);
        }
    }
#else
    void debug(const char * /*format*/, Args... /*args*/) const {
    }
#endif
};

struct shuffle_first {};
struct rndswap_first {};

//! Common form for all embedding problems.
//!
//! Needs to be extended with a fixed handler and domain handler to be complete.
class embedding_problem_base {
  protected:
    int num_v, num_f, num_q, num_r;

    //! Mutable references to qubit numbers and variable numbers
    vector<vector<int>> &qubit_nbrs, &var_nbrs;

    //! distribution over [0, 0xffffffff]
    uniform_int_distribution<> rand;

    vector<int> var_order_space;
    vector<int> var_order_visited;
    vector<int> var_order_shuffle;

    unsigned int exponent_margin;  // probably going to move this weight stuff out to another handler
  public:
    //! A mutable reference to the user specified parameters
    optional_parameters &params;

    double max_beta, round_beta, bound_beta;
    distance_t weight_table[64];

    int initialized, embedded, desperate, target_chainsize, improved, weight_bound;

    embedding_problem_base(optional_parameters &p_, int n_v, int n_f, int n_q, int n_r, vector<vector<int>> &v_n,
                           vector<vector<int>> &q_n)
            : num_v(n_v),
              num_f(n_f),
              num_q(n_q),
              num_r(n_r),
              qubit_nbrs(q_n),
              var_nbrs(v_n),
              rand(0, 0xffffffff),
              var_order_space(n_v),
              var_order_visited(n_v, 0),
              var_order_shuffle(n_v),
              exponent_margin(compute_margin()),
              params(p_) {
        if (exponent_margin <= 0) throw MinorMinerException("problem has too few nodes or edges");
        reset_mood();
    }

    virtual ~embedding_problem_base() {}

    //! resets some internal, ephemeral, variables to a default state
    void reset_mood() {
        auto ultramax_weight = 63. - std::log2(exponent_margin);

        if (ultramax_weight < 2) throw MinorMinerException("problem is too large to avoid overflow");

        if (ultramax_weight < params.max_fill)
            weight_bound = static_cast<int>(std::floor(ultramax_weight));
        else
            weight_bound = params.max_fill;

        max_beta = max(1., params.max_beta);
        round_beta = numeric_limits<double>::max();
        bound_beta = min(max_beta, exp2(ultramax_weight));
        initialized = embedded = desperate = target_chainsize = improved = 0;
    }

  private:
    //! computes an upper bound on the distances computed during tearout & replace
    size_t compute_margin() {
        if (num_q == 0) return 0;
        size_t max_degree =
                std::max_element(begin(var_nbrs), end(var_nbrs),
                                 [](const vector<int> &a, const vector<int> &b) { return a.size() < b.size(); })
                        ->size();
        if (max_degree == 0)
            return num_q;
        else
            return max_degree * num_q;
    }

  public:
    //! precomputes a table of weights corresponding to various overlap values `c`,
    //! for `c` from 0 to `max_weight`, inclusive.
    void populate_weight_table(int max_weight) {
        max_weight = min(63, max_weight);
        double log2base = (max_weight <= 0) ? 1 : ((63. - std::log2(exponent_margin)) / max_weight);
        double base = min(exp2(log2base), min(max_beta, round_beta));
        double power = 1;
        for (int i = 0; i <= max_weight; i++) {
            weight_table[i] = static_cast<distance_t>(power);
            power *= base;
        }
        for (int i = max_weight + 1; i < 64; i++) weight_table[i] = max_distance;
    }

    //! returns the precomputed weight associated with an overlap value of `c`
    distance_t weight(unsigned int c) const {
        if (c >= 64)
            return max_distance;
        else
            return weight_table[c];
    }

    //! a vector of neighbors for the variable `u`
    const vector<int> &var_neighbors(int u) const { return var_nbrs[u]; }

    //! a vector of neighbors for the variable `u`, pre-shuffling them
    const vector<int> &var_neighbors(int u, shuffle_first) {
        shuffle(std::begin(var_nbrs[u]), std::end(var_nbrs[u]));
        return var_nbrs[u];
    }

    //! a vector of neighbors for the variable `u`, applying a random
    //! transposition before returning the reference
    const vector<int> &var_neighbors(int u, rndswap_first) {
        if (var_nbrs[u].size() > 2) {
            size_t i = randint(0, var_nbrs[u].size() - 2);
            std::swap(var_nbrs[u][i], var_nbrs[u][i + 1]);
        } else if (var_nbrs[u].size() == 2) {
            if (randint(0, 1)) std::swap(var_nbrs[u][0], var_nbrs[u][1]);
        }
        return var_nbrs[u];
    }

    //! a vector of neighbors for the qubit `q`
    const vector<int> &qubit_neighbors(int q) const { return qubit_nbrs[q]; }

    //! number of variables which are not fixed
    inline int num_vars() const { return num_v; }

    //! number of qubits which are not reserved
    inline int num_qubits() const { return num_q; }

    //! number of fixed variables
    inline int num_fixed() const { return num_f; }

    //! number of reserved qubits
    inline int num_reserved() const { return num_r; }

    //! make a random integer between 0 and `m-1`
    int randint(int a, int b) { return rand(params.rng, typename decltype(rand)::param_type(a, b)); }

    //! shuffle the data bracketed by iterators `a` and `b`
    template <typename A, typename B>
    void shuffle(A a, B b) {
        std::shuffle(a, b, params.rng);
    }

    //! compute the connected component of the subset `component` of qubits,
    //! containing `q0`, and using`visited` as an indicator for which qubits
    //! have been explored
    void qubit_component(int q0, vector<int> &component, vector<int> &visited) {
        dfs_component(q0, qubit_nbrs, component, visited);
    }

    //! compute a variable ordering according to the `order` strategy
    const vector<int> &var_order(VARORDER order = VARORDER_SHUFFLE) {
        if (order == VARORDER_KEEP) {
            minorminer_assert(var_order_space.size() > 0);
            return var_order_space;
        }
        var_order_space.clear();
        var_order_shuffle.clear();
        for (int v = num_v; v--;) var_order_shuffle.push_back(v);
        shuffle(std::begin(var_order_shuffle), std::end(var_order_shuffle));
        if (order == VARORDER_SHUFFLE) {
            var_order_shuffle.swap(var_order_space);
        } else {
            var_order_visited.assign(num_v, 0);
            var_order_visited.resize(num_v + num_f, 1);
            for (auto v : var_order_shuffle)
                if (!var_order_visited[v]) switch (order) {
                        case VARORDER_DFS:
                            dfs_component(v, var_nbrs, var_order_space, var_order_visited);
                            break;
                        case VARORDER_BFS:
                            bfs_component(v, var_nbrs, var_order_space, var_order_visited, var_order_shuffle);
                            break;
                        case VARORDER_PFS:
                            pfs_component<min_queue<int>>(v, var_nbrs, var_order_space, var_order_visited,
                                                          var_order_shuffle);
                            break;
                        case VARORDER_RPFS:
                            pfs_component<max_queue<int>>(v, var_nbrs, var_order_space, var_order_visited,
                                                          var_order_shuffle);
                            break;
                        default:
                            // this should be unreachable...
                            throw CorruptParametersException("unsupported variable ordering specified");
                    }
        }
        return var_order_space;
    }

    //! Perform a depth first search
    void dfs_component(int x, const vector<vector<int>> &neighbors, vector<int> &component, vector<int> &visited) {
        size_t front = component.size();
        component.push_back(x);
        visited[x] = 1;
        while (front < component.size()) {
            int x = component[front++];
            size_t lastback = component.size();
            for (auto &y : neighbors[x]) {
                if (!visited[y]) {
                    visited[y] = 1;
                    component.push_back(y);
                }
            }
            if (lastback != component.size()) shuffle(std::begin(component) + lastback, std::end(component));
        }
    }

  private:
    //! Perform a priority first search (priority = #of visited neighbors)
    template <typename queue_t>
    void pfs_component(int x, const vector<vector<int>> &neighbors, vector<int> &component, vector<int> &visited,
                       vector<int> shuffled) {
        queue_t pq;
        pq.emplace(x, shuffled[x], 0);
        while (!pq.empty()) {
            auto z = pq.top();
            pq.pop();
            x = z.node;
            if (visited[x]) continue;
            visited[x] = 1;
            component.push_back(x);

            for (auto y : neighbors[x]) {
                if (!visited[y]) {
                    int d = 0;
                    for (auto w : neighbors[y]) d -= visited[w];
                    pq.emplace(y, shuffled[y], d);
                }
            }
        }
    }

    //! Perform a breadth first search, shuffling level sets
    void bfs_component(int x, const vector<vector<int>> &neighbors, vector<int> &component, vector<int> &visited,
                       vector<int> &shuffled) {
        min_queue<int> pq;
        pq.emplace(x, shuffled[x], 0);
        visited[x] = 1;
        while (!pq.empty()) {
            auto z = pq.top();
            pq.pop();
            x = z.node;
            auto d = z.dist;
            component.push_back(x);

            for (auto y : neighbors[x]) {
                if (!visited[y]) {
                    pq.emplace(y, shuffled[y], d + 1);
                    visited[y] = 1;
                }
            }
        }
    }
};

//! A template to construct a complete embedding problem by combining
//! `embedding_problem_base` with fixed/domain handlers.
template <class fixed_handler, class domain_handler, class output_handler>
class embedding_problem : public embedding_problem_base,
                          public fixed_handler,
                          public domain_handler,
                          public output_handler {
  private:
    using ep_t = embedding_problem_base;
    using fh_t = fixed_handler;
    using dh_t = domain_handler;
    using oh_t = output_handler;

  public:
    embedding_problem(optional_parameters &p, int n_v, int n_f, int n_q, int n_r, vector<vector<int>> &v_n,
                      vector<vector<int>> &q_n)
            : embedding_problem_base(p, n_v, n_f, n_q, n_r, v_n, q_n),
              fixed_handler(p, n_v, n_f, n_q, n_r),
              domain_handler(p, n_v, n_f, n_q, n_r),
              output_handler(p) {}
    virtual ~embedding_problem() {}
};
}  // namespace find_embedding
