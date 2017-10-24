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

// This file contains component classes for constructing embedding problems.  Presently, an embedding_problem class is
// constructed by combining the embedding_problem_base class with a domain_handler class and a fixed_handler class.
// This is used to accomplish dynamic dispatch for code in/around the inner loops without fouling their performance.

// Domain handlers are used to control which qubits are available for use for which variables.
// they perform three operations:
//   * marking unavailable qubits as "visited" so that we don't visit them in Dijkstra's algorithm
//   * setting unavailable qubits to an effectively infinite cost for root selection
//   * checking if a particular qubit is available for a particular variable

// this is the trivial domain handler, where every variable is allowed to use every qubit
class domain_handler_universe {
  public:
    domain_handler_universe(optional_parameters & /*p*/, int /*n_v*/, int /*n_f*/, int /*n_q*/, int /*n_r*/) {}

    static inline void prepare_visited(vector<int> &visited, int /*u*/, int /*v*/) {
        std::fill(begin(visited), end(visited), 0);
    }

    static inline void prepare_distances(vector<distance_t> &distance, const int /*u*/, const distance_t & /*mask_d*/) {
        std::fill(begin(distance), end(distance), 0);
    }

    static inline void prepare_distances(vector<distance_t> &distance, const int /*u*/, const distance_t & /*mask_d*/,
                                         const int start, const int stop) {
        std::fill(begin(distance) + start, begin(distance) + stop, 0);
    }

    static inline bool accepts_qubit(int /*u*/, int /*q*/) { return 1; }
};

// this domain handler stores masks for each variable so that prepare_visited and prepare_distances are barely more
// expensive than a memcopy
class domain_handler_masked {
    optional_parameters &params;
    vector<vector<int>> masks;

  public:
    domain_handler_masked(optional_parameters &p, int n_v, int n_f, int n_q, int n_r)
            : params(p), masks(n_v + n_f, vector<int>()) {
#ifndef NDEBUG
        for (auto &vC : params.restrict_chains)
            for (auto &q : vC.second) minorminer_assert(0 <= q && q < n_q + n_r);
#endif
        auto nostrix = end(params.restrict_chains);
        for (int v = n_v + n_f; v--;) {
            auto chain = params.restrict_chains.find(v);
            if (chain != nostrix) {
                masks[v].resize(n_q, -1);
                auto vC = *chain;
                for (auto &q : vC.second) masks[v][q] = 0;
            } else {
                masks[v].resize(n_q, 0);
            }
        }
    }

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

// This fixed handler is used when there are no fixed variables.
class fixed_handler_none {
  public:
    fixed_handler_none(optional_parameters & /*p*/, int /*n_v*/, int /*n_f*/, int /*n_q*/, int /*n_r*/) {}

    static inline bool fixed(int /*u*/) { return false; }

    static inline bool reserved(int /*u*/) { return false; }
};

// This fixed handler is used when the fixed variables are processed before instantiation and relabeled such that
// variables v >= num_v are fixed and qubits q >= num_q are reserved
class fixed_handler_hival {
  private:
    int num_v, num_q;

  public:
    fixed_handler_hival(optional_parameters & /*p*/, int n_v, int /*n_f*/, int n_q, int /*n_r*/)
            : num_v(n_v), num_q(n_q) {}
    inline bool fixed(const int u) { return u >= num_v; }

    inline bool reserved(const int q) { return q >= num_q; }
};

// This fixed handler is used when variables are allowed to be fixed after instantiation.  For that functionality, we
// probably need...
// * dynamic modification of var_neighbors and qubit_neighbors to maintain speed gains: fixed variables are sinks,
// reserved qubits are sources.
// * access to / ownership of var_neighbors and qubit_neighbors in this data structure
// * move existing initialization code from find_embedding.hpp into fixed_handler_hival (note the interplay with
// shuffling qubit labels, this might get gross)
class fixed_handler_list {
  private:
    vector<int> var_fixed;

  public:
    fixed_handler_list(optional_parameters &p, int n_v, int n_f, int /*n_q*/, int n_r) : var_fixed(n_v, 0) {
        minorminer_assert(n_f == 0);
        minorminer_assert(n_r == 0);
        for (auto &vC : p.fixed_chains) var_fixed[vC.first] = 1;
    }

    inline bool fixed(const int u) { return static_cast<bool>(var_fixed[u]); }

    inline bool reserved(const int) { return 0; }
};

// Common form for all embedding problems.
//
// Needs to be extended with a fixed handler and domain handler to be complete.
class embedding_problem_base {
  protected:
    int num_v, num_f, num_q, num_r;

    // Mutable references to qubit numbers and variable numbers
    vector<vector<int>> &qubit_nbrs, &var_nbrs;

    // A mutable reference to the user specified parameters
    optional_parameters &params;

    // distribution over [0, 0xffffffff]
    uniform_int_distribution<> rand;

    vector<int> var_order_space;
    vector<int> var_order_visited;
    vector<int> var_order_shuffle;

  public:
    int alpha, initialized, embedded, desperate, target_chainsize, weight_bound;

    embedding_problem_base(optional_parameters &p_, int n_v, int n_f, int n_q, int n_r, vector<vector<int>> &v_n,
                           vector<vector<int>> &q_n)
            : num_v(n_v),
              num_f(n_f),
              num_q(n_q),
              num_r(n_r),
              qubit_nbrs(q_n),
              var_nbrs(v_n),
              params(p_),
              rand(0, 0xffffffff),
              var_order_space(n_v),
              var_order_visited(n_v, 0),
              var_order_shuffle(n_v),
              initialized(0),
              embedded(0),
              desperate(0),
              target_chainsize(0) {
        alpha = 8 * sizeof(distance_t);
        int N = num_q;
        while (N /= 2) alpha--;
        weight_bound = min(params.max_fill, alpha);
    }

    const vector<int> &var_neighbors(int u) const { return var_nbrs[u]; }

    const vector<int> &qubit_neighbors(int u) const { return qubit_nbrs[u]; }

    inline int num_vars() const { return num_v; }
    inline int num_qubits() const { return num_q; }
    inline int num_fixed() const { return num_f; }
    inline int num_reserved() const { return num_r; }

    template <typename... Args>
    void display_message(const char *format, Args... args) const {
        if (params.verbose) {
            char buffer[1024];
            snprintf(buffer, 1024, format, args...);
            params.localInteractionPtr->displayOutput(buffer);
        }
    }

    int randint(int m) { return rand(params.rng, typename decltype(rand)::param_type(0, m - 1)); }

    template <typename A, typename B>
    void shuffle(A a, B b) {
        std::shuffle(a, b, params.rng);
    }

    const vector<int> &var_order() {
        var_order_space.clear();
        var_order_shuffle.clear();
        var_order_visited.assign(num_v, 0);
        var_order_visited.resize(num_v + num_f, 1);
        for (int v = num_v; v--;) var_order_shuffle.push_back(v);
        shuffle(begin(var_order_shuffle), end(var_order_shuffle));
        for (auto &v : var_order_shuffle)
            if (!var_order_visited[v]) dfs_component(v, var_nbrs, var_order_space, var_order_visited);
        return var_order_space;
    }

  private:
    // Perform a depth first search
    void dfs_component(int x, const vector<vector<int>> &neighbors, vector<int> &component, vector<int> &visited) {
        size_t front = component.size();
        component.push_back(x);
        visited[x] = 1;
        while (front < component.size()) {
            int x = component[front++];
            for (auto &y : neighbors[x]) {
                if (!visited[y]) {
                    visited[y] = 1;
                    component.push_back(y);
                }
            }
        }
    }
};

// A template to construct a complete embedding problem by combining
// `embedding_problem_base` with fixed/domain handlers.
template <class fixed_handler, class domain_handler>
class embedding_problem : public embedding_problem_base, public fixed_handler, public domain_handler {
  private:
    using ep_t = embedding_problem_base;
    using fh_t = fixed_handler;
    using dh_t = domain_handler;

  public:
    embedding_problem(optional_parameters &p, int n_v, int n_f, int n_q, int n_r, vector<vector<int>> &v_n,
                      vector<vector<int>> &q_n)
            : embedding_problem_base(p, n_v, n_f, n_q, n_r, v_n, q_n),
              fixed_handler(p, n_v, n_f, n_q, n_r),
              domain_handler(p, n_v, n_f, n_q, n_r) {}
};
}
