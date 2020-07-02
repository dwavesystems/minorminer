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
#include <iostream>
#include "util.hpp"

namespace find_embedding {

#ifdef CPPDEBUG
#define DIAGNOSE_CHAINS(other) \
    diagnostic();              \
    other.diagnostic();
#define DIAGNOSE_CHAIN() diagnostic();
#else
#define DIAGNOSE_CHAINS(other)
#define DIAGNOSE_CHAIN()
#endif

//! This class stores chains for embeddings, and performs qubit-use
//! accounting.  The `label` is the index number for the variable
//! represented by this chain.  The `links` member of a chain is an
//! unordered map storing the linking information for this chain.
//! The `data` member of a chain stores the connectivity information
//! for the chain.
//!
//! Links:
//! If `u` and `v` are variables which are connected by an edge,
//! the following must be true:
//!    either chain_u or chain_v is empty,
//!
//!    or
//!
//!    chain_u.links[v] is a key in chain_u.data,
//!    chain_v.links[u] is a key in chain_v.data, and
//!    (chain_u.links[v], chain_v.links[u]) are adjacent in the qubit graph
//!
//! Moreover, (chain_u.links[u]) must exist if chain_u is not empty,
//! and this is considered the root of the chain.
//!
//! Data:
//! The `data` member stores the connectivity information.  More
//! precisely, `data` is a mapping `qubit->(parent, refs)` where:
//!     `parent` is also contained in the chain
//!     `refs` is the total number of references to `qubit`, counting
//!            both parents and links
//!     the chain root is its own parent.

struct frozen_chain {
    unordered_map<int, pair<int, int>> data;
    unordered_map<int, int> links;
    void clear() {
        data.clear();
        links.clear();
    }
};

class chain {
  private:
    vector<int> &qubit_weight;
    unordered_map<int, pair<int, int>> data;
    unordered_map<int, int> links;
#ifdef CPPDEBUG
    bool belay_diagnostic;
#endif

  public:
    const int label;

    //! construct this chain, linking it to the qubit_weight vector `w` (common to
    //! all chains in an embedding, typically) and setting its variable label `l`
    chain(vector<int> &w, int l) : qubit_weight(w), data(), links(), label(l) {
#ifdef CPPDEBUG
        belay_diagnostic = false;
#endif
    }

    //! assign this to a vector of ints.  each incoming qubit will
    //! have itself as a parent.
    chain &operator=(const vector<int> &c) {
        clear();
        for (auto &q : c) {
            data.emplace(q, pair<int, int>(q, 1));
            minorminer_assert(0 <= q && q < static_cast<int>(qubit_weight.size()));
            qubit_weight[q]++;
        }
        DIAGNOSE_CHAIN();
        return *this;
    }

    //! assign this to another chain
    chain &operator=(const chain &c) {
        clear();
        data = c.data;
        for (auto &q : c) qubit_weight[q]++;
        links = c.links;
        DIAGNOSE_CHAIN();
        return *this;
    }

    //! number of qubits in chain
    inline size_t size() const { return data.size(); }

    //! returns 0 if `q` is not contained in `this`, 1 otherwise
    inline size_t count(const int q) const { return data.count(q); }

    //! get the qubit, in `this`, which links `this` to the chain of x
    //!(if x==label, interpret the linking qubit as the chain's root)
    inline int get_link(const int x) const {
        auto z = links.find(x);
        if (z == links.end())
            return -1;
        else
            return (*z).second;
    }

    //! set the qubit, in `this`, which links `this` to the chain of x
    //!(if x==label, interpret the linking qubit as the chain's root)
    inline void set_link(const int x, const int q) {
        minorminer_assert(get_link(x) == -1);
        minorminer_assert(count(q) == 1);
        links[x] = q;

        retrieve(q).second++;
        DIAGNOSE_CHAIN();
    }

    //! discard and return the linking qubit for `x`, or -1 if that link is not set
    inline int drop_link(const int x) {
        int q = -1;
        auto z = links.find(x);
        if (z != links.end()) {
            q = (*z).second;
            minorminer_assert(count(q) == 1);
            retrieve(q).second--;
            links.erase(z);
        }
        DIAGNOSE_CHAIN();
        return q;
    }

    //! insert the qubit `q` into `this`, and set `q` to be the root
    //!(represented as the linking qubit for `label`)
    inline void set_root(const int q) {
        minorminer_assert(data.size() == 0);
        minorminer_assert(links.size() == 0);
        links.emplace(label, q);
        data.emplace(q, pair<int, int>(q, 2));
        qubit_weight[q]++;
        DIAGNOSE_CHAIN();
    }

    //! empty this data structure
    inline void clear() {
        for (auto &q : *this) qubit_weight[q]--;
        data.clear();
        links.clear();
        DIAGNOSE_CHAIN();
    }

    //! add the qubit `q` as a leaf, with `parent` as its parent
    inline void add_leaf(const int q, const int parent) {
        minorminer_assert(data.count(q) == 0);
        minorminer_assert(data.count(parent) == 1);
        data.emplace(q, pair<int, int>(parent, 0));
        qubit_weight[q]++;
        retrieve(parent).second++;
        DIAGNOSE_CHAIN();
    }

    //! try to delete the qubit `q` from this chain, and keep
    //! deleting until no more qubits are free to be deleted.
    //! return the first ancestor which cannot be deleted
    inline int trim_branch(int q) {
        minorminer_assert(data.count(q) == 1);
        int p = trim_leaf(q);
        while (p != q) {
            q = p;
            p = trim_leaf(q);
        }
        minorminer_assert(data.count(q) == 1);
        DIAGNOSE_CHAIN();
        return q;
    }

    //! try to delete the qubit `q` from this chain.  if `q`
    //! cannot be deleted, return it; otherwise return its parent
    inline int trim_leaf(int q) {
        minorminer_assert(data.count(q) == 1);
        auto z = data.find(q);
        auto p = (*z).second;
        if (p.second == 0) {
            qubit_weight[q]--;
            retrieve(p.first).second--;
            data.erase(z);
            q = p.first;
        }
        DIAGNOSE_CHAIN();
        return q;
    }

    //! the parent of `q` in this chain -- which might be `q` but
    //! otherwise cycles should be impossible
    inline int parent(const int q) const {
        minorminer_assert(data.count(q) == 1);
        return fetch(q).first;
    }

    //! assign `p` to be the parent of `q`, on condition that both `p` and `q`
    //! are contained in `this`, `q` is its own parent, and `q` is not the root
    inline void adopt(const int p, const int q) {
        minorminer_assert(data.count(q) == 1);
        minorminer_assert(data.count(p) == 1);
        auto &P = retrieve(p);
        auto &Q = retrieve(q);
        minorminer_assert(Q.first == q);
        minorminer_assert(get_link(label) != q);
        Q.first = p;
        Q.second--;
        P.second++;
        minorminer_assert(parent(q) == p);
        DIAGNOSE_CHAIN();
    }

    //! return the number of references that `this` makes to the qubit
    //!`q` -- where a "reference" is an occurrence of `q` as a parent
    //! or an occurrence of `q` as a linking qubit / root
    inline int refcount(const int q) const {
        minorminer_assert(data.count(q) == 1);
        return fetch(q).second;
    }

    //! store this chain into a `frozen_chain`, unlink all chains from
    //! this, and clear()
    inline size_t freeze(vector<chain> &others, frozen_chain &keep) {
        keep.clear();
        for (auto &v_p : links) {
            keep.links.emplace(v_p);
            int v = v_p.first;
            if (v != label) {
                minorminer_assert(0 <= v && v < static_cast<int>(others.size()));
                int q = others[v].drop_link(label);
                keep.links.emplace(-v - 1, q);
            }
        }
        links.clear();
        for (auto &q : *this) qubit_weight[q]--;
        keep.data.swap(data);
        minorminer_assert(size() == 0);
        DIAGNOSE_CHAIN();
        return keep.data.size();
    }

    //! restore a `frozen_chain` into this, re-establishing links
    //! from other chains.  precondition: this is empty.
    inline void thaw(vector<chain> &others, frozen_chain &keep) {
        minorminer_assert(size() == 0);
        keep.data.swap(data);
        for (auto &q : *this) qubit_weight[q]++;
        for (auto &v_p : keep.links) {
            int v = v_p.first;
            if (v >= 0) {
                links.emplace(v_p);
            } else {
                v = -v - 1;
                minorminer_assert(0 <= v && v < static_cast<int>(others.size()));
                others[v].set_link(label, v_p.second);
            }
        }
        DIAGNOSE_CHAIN();
    }

    //! assumes `this` and `other` have links for eachother's labels
    //! steals all qubits from `other` which are available to be taken
    //! by `this`; starting with the qubit links and updating qubit
    //! links after all
    template <typename embedding_problem_t>
    inline void steal(chain &other, embedding_problem_t &ep, int chainsize = 0) {
        int q = drop_link(other.label);
        int p = other.drop_link(label);

        minorminer_assert(q != -1);
        minorminer_assert(p != -1);

        while ((chainsize == 0 || static_cast<int>(size()) < chainsize) && ep.accepts_qubit(label, p)) {
            int r = other.trim_leaf(p);
            minorminer_assert(other.size() >= 1);
            if (r == p) break;
            auto z = data.find(p);
            if (z == data.end())
                add_leaf(p, q);
            else if (p != q) {
                auto &w = (*z).second;
                w.second++;
#ifdef CPPDEBUG
                belay_diagnostic = true;
#endif
                trim_branch(q);
#ifdef CPPDEBUG
                belay_diagnostic = false;
#endif
                w.second--;
            }
            q = p;
            p = r;
        }

        minorminer_assert(other.count(p) == 1);
        minorminer_assert(count(q) == 1);

        set_link(other.label, q);
        other.set_link(label, p);
        DIAGNOSE_CHAINS(other);
    }

    //! link this chain to another, following the path
    //!   `q`, `parent[q]`, `parent[parent[q]]`, ...
    //! from `this` to `other` and intermediate nodes (all but the last)
    //! into `this` (preconditions: `this` and `other` are not linked,
    //! `q` is contained in `this`, and the parent-path is eventually
    //! contained in `other`)
    void link_path(chain &other, int q, const vector<int> &parents) {
        minorminer_assert(count(q) == 1);
        minorminer_assert(links.count(other.label) == 0);
        minorminer_assert(other.get_link(label) == -1);
        int p = parents[q];
        if (p == -1) {
            p = q;
        } else {
            while (other.count(p) == 0) {
                if (count(p))
                    trim_branch(q);
                else
                    add_leaf(p, q);
                q = p;
                p = parents[p];
            }
        }
        minorminer_assert(other.count(p) == 1);
        minorminer_assert(count(q) == 1);
        set_link(other.label, q);
        other.set_link(label, p);
        DIAGNOSE_CHAINS(other);
    }

    class iterator {
      public:
        iterator(typename decltype(data)::const_iterator it) : it(it) {}
        iterator operator++() { return ++it; }
        bool operator!=(const iterator &other) { return it != other.it; }
        const typename decltype(data)::key_type &operator*() const { return it->first; }  // Return key part of map

      private:
        typename decltype(data)::const_iterator it;
    };

    //! iterator pointing to the first qubit in this chain
    iterator begin() const { return iterator(data.begin()); }

    //! iterator pointing to the end of this chain
    iterator end() const { return iterator(data.end()); };

    //! run the diagnostic, and if it fails, report the failure to the user
    //! and throw a CorruptEmbeddingException.  the `last_op` argument is
    //! used in the error message
    inline void diagnostic() {
#ifdef CPPDEBUG
        if (belay_diagnostic) return;
#endif
        switch (run_diagnostic()) {
            case 1:
                throw CorruptEmbeddingException("Qubit containment errors found in chain diagnostics.");
            case 2:
                throw CorruptEmbeddingException("Parent errors found in chain diagnostics.");
            case 3:
                throw CorruptEmbeddingException("Multiple errors found in chain diagnostics.");
        }
    }

    //! run the diagnostic and return a nonzero status `r` in case of failure
    //! if(`r`&1), then the parent of a qubit is not contained in this chain
    //! if(`r`&2), then there is a refcounting error in this chain
    inline int run_diagnostic() const {
        int errorcode = 0;
        unordered_map<int, int> refs;
        for (auto &x_q : links) {
            refs[x_q.second]++;
        }
        for (auto &q_pr : data) {
            refs[q_pr.second.first]++;
        }
        for (auto &p_r : refs) {
            int p = p_r.first;
            int r = p_r.second;
            if (count(p) == 0)
                errorcode |= 1;
            else if (r != data.at(p).second)
                errorcode |= 2;
        }
        return errorcode;
    }

  private:
    //! const unsafe data accessor
    inline const pair<int, int> &fetch(int q) const { return (*data.find(q)).second; }

    //! non-const unsafe data accessor
    inline pair<int, int> &retrieve(int q) { return (*data.find(q)).second; }
};
}  // namespace find_embedding
