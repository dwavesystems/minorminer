#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "chain.hpp"
#include "embedding_problem.hpp"
#include "pairing_queue.hpp"
#include "util.hpp"

namespace find_embedding {

// This class is how we represent and manipulate embedding objects, using as
// much encapsulation as possible.  We provide methods to view and modify
// chains.

#ifdef CPPDEBUG
#define DIAGNOSE(X) long_diagnostic(X);
#else
#define DIAGNOSE(X)
#endif

template <typename embedding_problem_t>
class embedding {
  private:
    embedding_problem_t &ep;
    int num_qubits, num_reserved;
    int num_vars, num_fixed;

    // this is where we store chains -- see chain.hpp for how
    vector<chain> var_embedding;

#ifdef CPPDEBUG
    char *last_diagnostic;
#endif

    // weights, that is, the number of non-fixed chains that use each qubit
    // this is used in pathfinder clases to determine non-overlapped, or
    // or least-overlapped paths through the qubit graph
    vector<int> qub_weight;

  public:
    embedding(embedding_problem_t &e_p)
            : ep(e_p),
              num_qubits(ep.num_qubits()),
              num_reserved(ep.num_reserved()),
              num_vars(ep.num_vars()),
              num_fixed(ep.num_fixed()),
              var_embedding(),
#ifdef CPPDEBUG
              last_diagnostic(nullptr),
#endif
              qub_weight(num_qubits, 0) {
        for (int q = 0; q < num_vars + num_fixed; q++) var_embedding.emplace_back(q);
        DIAGNOSE("post base_construct");
    }

    embedding(embedding_problem_t &e_p, map<int, vector<int>> &fixed_chains, map<int, vector<int>> &initial_chains)
            : embedding(e_p) {
        for (auto &vC : fixed_chains) fix_chain(vC.first, vC.second);

        for (auto &vC : initial_chains)
            if (!ep.fixed(vC.first)) set_chain(vC.first, vC.second);

        for (auto &vC : initial_chains) {
            var_embedding[vC.first].set_link(vC.first, vC.second[0]);
            for (auto &u : ep.var_neighbors(vC.first))
                if (u > vC.first) linkup(vC.first, u);
        }
        DIAGNOSE("post construct");
    }

    embedding<embedding_problem_t> &operator=(const embedding<embedding_problem_t> &other) {
        if (this != &other) {
            qub_weight = other.qub_weight;
            var_embedding = other.var_embedding;
        }
        return *this;
    }

    // Get the variables in a chain
    inline const chain &get_chain(int v) const { return var_embedding[v]; }

    // Get the size of a chain
    inline int chainsize(int v) const { return var_embedding[v].size(); }

    // Get the weight of a qubit
    inline int weight(int q) const { return qub_weight[q]; }

    // Get the maximum of all qubit weights
    inline int max_weight() const { return *max_element(begin(qub_weight), end(qub_weight)); }

    // Get the maximum of all qubit weights in a range
    inline int max_weight(const int start, const int stop) const {
        return *max_element(begin(qub_weight) + start, begin(qub_weight) + stop);
    }

    // Check if variable v is includes qubit q in its chain
    inline bool has_qubit(const int v, const int q) const { return static_cast<bool>(var_embedding[v].count(q)); }

    // Assign a chain for variable u
    inline void set_chain(const int u, const vector<int> &incoming) {
        // remove the current chain and account for its qubits
        for (auto &q : var_embedding[u]) {
            qub_weight[q]--;
        }
        var_embedding[u] = incoming;
        for (auto &q : var_embedding[u]) {
            qub_weight[q]++;
        }
        DIAGNOSE("set_chain");
    }

    // Permanently assign a chain for variable u.
    // NOTE: This must be done before any chain is assigned to u.
    inline void fix_chain(const int u, const vector<int> &incoming) {
// assume this embedding doesn't have a chain for u yet
#ifdef CPPDEBUG
        minorminer_assert(chainsize(u) == 0);
        minorminer_assert(ep.fixed(u));
        for (auto &q : incoming) {
            minorminer_assert(ep.reserved(q));
        }
#endif
        var_embedding[u] = incoming;
        DIAGNOSE("fix_chain");
    }

    inline bool operator==(const embedding &b) const {
        for (int v = num_vars; v--;)
            for (int q = num_qubits; q--;)
                if (has_qubit(v, q) != b.has_qubit(v, q)) return false;
        return true;
    }

    void construct_chain(const int u, const int q, const int target_chainsize, const vector<vector<int>> &parents) {
        var_embedding[u].set_root(q);

        // extract the paths from each parents list
        for (auto &v : ep.var_neighbors(u))
            if (chainsize(v)) extract_path(u, v, q, parents[v]);

        // account for the weight of new qubits
        for (auto &q : var_embedding[u]) qub_weight[q]++;

        // distribute path segments to the neighboring chains -- path segments are the qubits
        // that are ONLY used to join link_qubit[u][v] to link_qubit[u][u] and aren't used
        // for any other variable
        // * if the target chainsize is zero, dump the entire segment into the neighbor
        // * if the target chainsize is k, dump the largest portion of the segment
        for (auto &v : ep.var_neighbors(u))
            if (chainsize(v) && !(ep.fixed(v))) var_embedding[v].steal(var_embedding[u], ep, target_chainsize);

        DIAGNOSE("construct_chain")
    }

    void tear_out(int u) {
        // short tearout procedure
        // blank out the chain, its linking qubits, and account for the qubits being freed
        for (auto &q : var_embedding[u]) qub_weight[q]--;
        var_embedding[u].clear();
        for (auto &v : ep.var_neighbors(u)) var_embedding[v].drop_link(u);
        DIAGNOSE("tear_out")
    }

    void covfefe(int u) {
        // grow the chain for `u`, stealing all available qubits from neighboring variables
        for (auto &q : var_embedding[u]) qub_weight[q]--;
        for (auto &v : ep.var_neighbors(u)) {
            if (ep.fixed(v)) continue;
            if (var_embedding[u].get_link(v) == -1) continue;
            if (var_embedding[v].get_link(u) == -1) continue;
            for (auto &q : var_embedding[v]) qub_weight[q]--;
            var_embedding[u].steal(var_embedding[v], ep);
            for (auto &q : var_embedding[v]) qub_weight[q]++;
        }
        for (auto &q : var_embedding[u]) qub_weight[q]++;
        DIAGNOSE("covfefe")
    }

    void statistics(int &embeddingSum, int &bagWidth, int &numMaxBags, int &maxChainSize, int &numMaxChains) const {
        embeddingSum = bagWidth = numMaxBags = maxChainSize = numMaxChains = 0;
        for (int q = num_qubits; q--;) {
            if (qub_weight[q] > bagWidth) {
                bagWidth = qub_weight[q];
                numMaxBags = 1;
            } else if (qub_weight[q] == bagWidth) {
                numMaxBags++;
            }
        }
        for (int v = num_vars; v--;) {
            int s = chainsize(v);
            embeddingSum += s;
            if (chainsize(v) > maxChainSize) {
                maxChainSize = s;
                numMaxChains = 1;
            } else if (s == maxChainSize) {
                numMaxChains++;
            }
        }
    }

    // check if the embedding is fully linked -- that is, if each pair of adjacent
    // variables is known to correspond to a pair of adjacent qubits
    bool linked() const {
        for (int u = num_vars; u--;)
            if (!linked(u)) return false;
        return true;
    }

    // check if a single variable is linked with all adjacent variables.
    bool linked(int u) const {
        if (var_embedding[u].get_link(u) < 0) return false;
        for (auto &v : ep.var_neighbors(u))
            if (var_embedding[u].get_link(v) < 0) return false;
        return true;
    }

  private:
    // This method attempts to find the linking qubits for a pair of adjacent variables, and
    // returns true/false on success/failure in finding that pair.
    bool linkup(int u, int v) {
        if ((var_embedding[u].get_link(v) >= 0) && (var_embedding[v].get_link(u) >= 0)) {
            return true;
        }
        for (auto &q : var_embedding[v]) {  // hax!  this plays nicely with reserved qubits being sources
            for (auto &p : ep.qubit_neighbors(q)) {
                if (has_qubit(u, p)) {
                    var_embedding[u].set_link(v, p);
                    var_embedding[v].set_link(u, q);
                    return true;
                }
            }
        }
        for (auto &q : var_embedding[u]) {
            if (has_qubit(v, q)) {
                var_embedding[u].set_link(v, q);
                var_embedding[v].set_link(u, q);
                return true;
            }
        }
        return false;
    }

    inline void extract_path(const int u, const int v, int curr_q, const vector<int> &parent) {
        // read off the path from u to v, starting from curr_q and
        // following parent until we fall off the end -- typically,
        // curr_q will be in the chain of u and "the end" is in the
        // chain of v
        minorminer_assert(curr_q >= 0);
        int prev_q = curr_q;
        int next_q = parent[curr_q];

        while (next_q != -1) {
            if (has_qubit(u, curr_q))
                var_embedding[u].trim_branch(prev_q);
            else
                var_embedding[u].add_leaf(curr_q, prev_q);

            prev_q = curr_q;
            curr_q = next_q;
            next_q = parent[curr_q];
        }

        minorminer_assert(has_qubit(u, prev_q));
        minorminer_assert(has_qubit(v, curr_q));
        var_embedding[u].set_link(v, prev_q);
        var_embedding[v].set_link(u, curr_q);
    }

  public:
    void print() const {
        ep.display_message("var_embedding = [");
        for (int u = 0; u < num_vars; u++) {
            ep.display_message("[");
            for (auto &q : var_embedding[u])
                ep.display_message("%d:(%d,%d),", q, var_embedding[u].parent(q), var_embedding[u].refcount(q));
            ep.display_message("],");
        }
        ep.display_message("]\n");

        ep.display_message("var_edges = {");
        for (int u = 0; u < num_vars + num_fixed; u++) {
            for (auto &v : ep.var_neighbors(u))
                if (var_embedding[u].get_link(v) >= 0)
                    ep.display_message("(%d,%d):%d,", u, v, var_embedding[u].get_link(v));
        }
        ep.display_message("}\n");

        ep.display_message("var_roots = {");
        for (int u = 0; u < num_vars; u++)
            if (var_embedding[u].get_link(u) >= 0) ep.display_message("%d:%d,", u, var_embedding[u].get_link(u));
        ep.display_message("}\n");
    }

    void long_diagnostic(char *current_state) {
        run_long_diagnostic(current_state);
#ifdef CPPDEBUG
        last_diagnostic = current_state;
#endif
    }

    void run_long_diagnostic(char *current_state) const {
        int err = 0;
        vector<int> tmp_weight;
        tmp_weight.assign(num_qubits + num_reserved, 0);
        int zeros = 0;
        int bad_parents = false;
        for (int v = 0; v < num_vars + num_fixed; v++) {
            for (auto &q : var_embedding[v]) {
                if (!ep.fixed(v)) {
                    tmp_weight[q]++;
                    auto z = var_embedding[v].parent(q);
                    if (z != q) {
                        bool got = false;
                        for (auto &p : ep.qubit_neighbors(q))
                            if (p == z) got = true;
                        for (auto &p : ep.qubit_neighbors(z))
                            if (p == q) got = true;
                        if (!got) {
                            ep.display_message("parent of qubit %d in chain %d is not a neighbor of %d\n", q, v, q);
                            err = 1;
                        }
                    }
                }
            }
            int errcode = var_embedding[v].run_diagnostic();
            if (!errcode) {
                for (auto &q : var_embedding[v]) {
                    int n = num_qubits + 1;
                    int z = var_embedding[v].parent(q);
                    int last_z = q;
                    while (z != last_z && n--) {
                        last_z = z;
                        z = var_embedding[v].parent(z);
                    }
                    if (n < 0) {
                        ep.display_message("cycle detected in parents for %d, entry point is %d\n", v, q);
                        err = 1;
                    }
                }
            } else {
                ep.display_message("chain datastructure invalid for %d\n", v);
                err = 1;
            }
            if (!var_embedding[v].size()) zeros++;
        }

        if (zeros > 1 && ep.initialized) {
            ep.display_message(
                    "more than one (%d) chains empty after initialization (should be 0 unless we've just torn a "
                    "variable out, thence 1)\n",
                    zeros);
            err = 1;
        }
        for (int v = num_vars + num_fixed; v--;) {
            int n = chainsize(v);
            vector<int> good_links;
            good_links.assign(num_vars + num_fixed, 0);
            if (chainsize(v)) {
                for (auto &u : ep.var_neighbors(v)) {
                    int link_u = var_embedding[u].get_link(v);
                    int link_v = var_embedding[v].get_link(u);
                    if (!chainsize(u)) {
                        if (link_u != -1) {
                            ep.display_message(
                                    "link qubit for problem interaction (%d -> %d) is set but %d's chain is empty\n", u,
                                    v, u);
                            err = 1;
                        }
                        if (link_v != -1) {
                            ep.display_message(
                                    "link qubit for problem interaction (%d -> %d) is set but %d's chain is empty\n", v,
                                    u, v);
                            err = 1;
                        }
                        continue;
                    }
                    int link = 0;
                    if (link_v == -1) {
                        if (ep.initialized) {
                            ep.display_message("link qubit for problem interaction (%d -> %d) is not set\n", v, u);
                            err = 1;
                        }
                    } else if (!has_qubit(v, link_v)) {
                        ep.display_message(
                                "link qubit for problem interaction (%d -> %d) is not present in chain for %d\n", v, u,
                                v);
                        err = 1;
                    } else
                        link++;

                    if (link_u == -1) {
                        if (ep.initialized) {
                            ep.display_message("link qubit for problem interaction (%d -> %d) is not set\n", u, v);
                            err = 1;
                        }
                    } else if (!has_qubit(u, link_u)) {
                        ep.display_message(
                                "link qubit for problem interaction (%d -> %d) is not present in chain for %d\n", u, v,
                                u);
                        err = 1;
                    } else
                        link++;

                    if (link == 2) {
                        int t = 0;
                        for (auto &z : ep.qubit_neighbors(link_v))
                            if (z == link_u) t = 1;
                        for (auto &z : ep.qubit_neighbors(link_u))
                            if (z == link_v) t = 1;

                        if (!t && link_u != link_v) {
                            ep.display_message(
                                    "link for problem interaction (%d,%d) uses coupler (%d,%d) which does not exist\n",
                                    u, v, link_v, link_u);
                            err = 1;
                        } else {
                            good_links[u] = 1;
                        }
                    }
                }
                int root = var_embedding[v].get_link(v);
                bool rooted = (root > -1);
                if (rooted) {
                    if (!ep.fixed(v)) {
                        vector<int> component;
                        component.push_back(root);
                        vector<int> visited;
                        visited.assign(num_qubits, 0);
                        visited[root] = 1;
                        unsigned int front = 0;
                        while (front < component.size()) {
                            int q = component[front++];
                            for (auto &p : ep.qubit_neighbors(q)) {
                                if (!visited[p] && has_qubit(v, p)) {
                                    visited[p] = 1;
                                    component.push_back(p);
                                }
                            }
                        }
                        if (chainsize(v) != component.size()) {
                            ep.display_message("chain for %d is disconnected but qubit root is set\n", v);
                            err = 1;
                        }
                    }
                }
            }
        }

        for (int q = num_qubits; q--;) {
            if (tmp_weight[q] != qub_weight[q]) {
                ep.display_message("qubit weight is out of date for %d (truth is %d, memo is %d)\n", q, tmp_weight[q],
                                   qub_weight[q]);
                err = 1;
            }
            if (ep.embedded && tmp_weight[q] > 1) {
                ep.display_message("qubit %d is overlapped after embedding success\n", q);
                err = 1;
            }
        }

        if (err) {
#ifdef CPPDEBUG
            if (last_diagnostic != nullptr) ep.display_message("last state was %s\n", last_diagnostic);
#endif
            ep.display_message("errors found in data structure, current state is '%s'.  quitting\n", current_state);
            print();
            std::flush(std::cout);
            throw - 1;
        }
    }
};
}
