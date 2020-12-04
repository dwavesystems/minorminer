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

#include <map>
#include <random>
#include <set>
#include <vector>
#include "util.hpp"

namespace graph {

template <typename T>
class unaryint {
  public:
};

template <>
class unaryint<bool> {
    const bool b;

  public:
    unaryint(const bool x) : b(x) {}
    int operator()(int) const { return b; }
};

template <>
class unaryint<std::vector<int>> {
    const std::vector<int> b;

  public:
    unaryint(const std::vector<int> m) : b(m) {}
    int operator()(int i) const { return b[i]; }
};

//! this one is a little weird -- construct a unaryint(nullptr)
//! and get back the identity function f(x) -> x
template <>
class unaryint<void*> {
  public:
    unaryint(void* const&) {}
    int operator()(int i) const { return i; }
};

template <>
class unaryint<int> {
    const int b;

  public:
    unaryint(int m) : b(m) {}
    int operator()(int i) const { return i >= b; }
};

//! Represents an undirected graph as a list of edges.
//!
//! Provides methods to extract those edges into neighbor lists (with options
//! to relabel and produce directed graphs).
//!
//! As an input to the library this may be a disconnected graph,
//! but when returned from components it is a connected sub graph.
class input_graph {
  private:
    // In
    std::vector<int> edges_aside;
    std::vector<int> edges_bside;
    size_t _num_nodes;

    //! this method converts a std::vector of sets into a std::vector of sets, ensuring
    //! that element i is not contained in nbrs[i].  this method is called by
    //! methods which produce neighbor sets (killing parallel/overrepresented
    //! edges), in order to kill self-loops and also store each neighborhood
    //! in a contiguous memory segment.
    std::vector<std::vector<int>> _to_vectorhoods(std::vector<std::set<int>>& _nbrs) const {
        std::vector<std::vector<int>> nbrs;
        for (size_t i = 0; i < _num_nodes; i++) {
            std::set<int>& nbrset = _nbrs[i];
            nbrset.erase(i);
            nbrs.emplace_back(std::begin(nbrset), std::end(nbrset));
        }
        return nbrs;
    }

  public:
    //! Constructs an empty graph.
    input_graph() : edges_aside(), edges_bside(), _num_nodes(0) {}
    //! Constructs a graph from the provided edges.
    //! The ends of edge ii are aside[ii] and bside[ii].
    //! @param n_v Number of nodes in the graph.
    //! @param aside List of nodes describing edges.
    //! @param bside List of nodes describing edges.
    input_graph(int n_v, const std::vector<int>& aside, const std::vector<int>& bside)
            : edges_aside(aside), edges_bside(bside), _num_nodes(n_v) {
        minorminer_assert(aside.size() == bside.size());
    }

    //! Remove all edges and nodes from a graph.
    void clear() {
        edges_aside.clear();
        edges_bside.clear();
        _num_nodes = 0;
    }

    //! Return the nodes on either end of edge `i`
    int a(const int i) const { return edges_aside[i]; }
    //! Return the nodes on either end of edge `i`
    int b(const int i) const { return edges_bside[i]; }

    //! Return the size of the graph in nodes
    size_t num_nodes() const { return _num_nodes; }
    //! Return the size of the graph in edges
    size_t num_edges() const { return edges_aside.size(); }

    //! Add an edge to the graph
    void push_back(int ai, int bi) {
        edges_aside.push_back(ai);
        edges_bside.push_back(bi);
        _num_nodes = std::max(_num_nodes, static_cast<size_t>(std::max(ai, bi) + 1));
    }

  private:
    //! produce the node->nodelist mapping for our graph, where certain nodes are
    //! marked as sources (no incoming edges), relabeling all nodes along the way,
    //! and filtering according to a mask.  note that the mask itself is assumed
    //! to be a union of components -- only one side of each edge is checked
    template <typename T1, typename T2, typename T3, typename T4>
    inline std::vector<std::vector<int>> __get_neighbors(const unaryint<T1>& sources, const unaryint<T2>& sinks,
                                                         const unaryint<T3>& relabel, const unaryint<T4>& mask) const {
        std::vector<std::set<int>> _nbrs(_num_nodes);
        for (size_t i = num_edges(); i--;) {
            int ai = a(i), bi = b(i);
            if (mask(ai)) {
                int rai = relabel(ai), rbi = relabel(bi);
                if (!sources(bi) && !sinks(ai)) _nbrs[rai].insert(rbi);
                if (!sources(ai) && !sinks(bi)) _nbrs[rbi].insert(rai);
            }
        }
        return _to_vectorhoods(_nbrs);
    }

    //! smash the types through unaryint
    template <typename T1, typename T2, typename T3 = void*, typename T4 = bool>
    inline std::vector<std::vector<int>> _get_neighbors(const T1& sources, const T2& sinks, const T3& relabel = nullptr,
                                                        const T4& mask = true) const {
        return __get_neighbors(unaryint<T1>(sources), unaryint<T2>(sinks), unaryint<T3>(relabel), unaryint<T4>(mask));
    }

  public:
    //! produce a std::vector<std::vector<int>> of neigborhoods, with certain nodes marked as sources (inbound edges are
    //! omitted)
    //! sources is either a std::vector<int> (where non-sources x have sources[x] = 0), or another type for which we
    //! have a
    //! unaryint specialization
    //! optional arguments: relabel, mask (any type with a unaryint specialization)
    //!    relabel is applied to the nodes as they are placed into the neighborhood list (and not used for checking
    //!    sources / mask)
    //!    mask is used to filter down to the induced graph on nodes x with mask[x] = 1
    template <typename T1, typename... Args>
    std::vector<std::vector<int>> get_neighbors_sources(const T1& sources, Args... args) const {
        return _get_neighbors(sources, false, args...);
    }

    //! produce a std::vector<std::vector<int>> of neigborhoods, with certain nodes marked as sinks (outbound edges are
    //! omitted)
    //! sinks is either a std::vector<int> (where non-sinks x have sinks[x] = 0), or another type for which we have a
    //! unaryint specialization
    //! optional arguments: relabel, mask (any type with a unaryint specialization)
    //!    relabel is applied to the nodes as they are placed into the neighborhood list (and not used for checking
    //!    sinks / mask)
    //!    mask is used to filter down to the induced graph on nodes x with mask[x] = 1
    template <typename T2, typename... Args>
    std::vector<std::vector<int>> get_neighbors_sinks(const T2& sinks, Args... args) const {
        return _get_neighbors(false, sinks, args...);
    }

    //! produce a std::vector<std::vector<int>> of neigborhoods
    //! optional arguments: relabel, mask (any type with a unaryint specialization)
    //!    relabel is applied to the nodes as they are placed into the neighborhood list (and not used for checking
    //!    mask)
    //!    mask is used to filter down to the induced graph on nodes x with mask[x] = 1
    template <typename... Args>
    std::vector<std::vector<int>> get_neighbors(Args... args) const {
        return _get_neighbors(false, false, args...);
    }
};

//! Represents a graph as a series of connected components.
//!
//! The input graph may consist of many components, they will be separated
//! in the construction.
class components {
  public:
    template <typename T>
    components(const input_graph& g, const unaryint<T>& reserve)
            : index(g.num_nodes(), 0), label(g.num_nodes(), 0), component(g.num_nodes()), component_g() {
        /*
        STEP 1: perform union/find to compute components.

        During this stage, we use this.index and this.label, respectively,
        to store the parent and rank data for union/find operations.
        */
        std::vector<int>& parent = index;
        for (size_t x = g.num_nodes(); x--;) {
            parent[x] = x;
        }
        for (size_t i = g.num_edges(); i--;) {
            __init_union(g.a(i), g.b(i));
        }

        for (size_t x = g.num_nodes(); x--;) component[__init_find(x)].push_back(x);

        sort(std::begin(component), std::end(component),
             [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() > b.size(); });

        /*
        STEP 2: distribute edges to their components

        Now, all component information is contained in this.component, so
        we're free to overwrite the data left in this.label and this.index.
        The labels associated with component[c] are the numbers 0 through
        component[c].size()-1.
        */
        for (size_t c = 0; c < g.num_nodes(); c++) {
            std::vector<int>& comp = component[c];
            auto back = std::end(comp);
            for (auto front = std::begin(comp); front < back; front++)
                while (front < back && reserve(*front)) iter_swap(front, --back);
            if (comp.size()) {
                for (size_t j = comp.size(); j--;) {
                    label[comp[j]] = j;
                    index[comp[j]] = c;
                }
                component_g.push_back(input_graph());
                _num_reserved.push_back(std::end(comp) - back);
            } else {
                component.resize(c);
                break;
            }
        }
        for (size_t i = g.num_edges(); i--;) {
            int a = g.a(i);
            int b = g.b(i);
            component_g[index[a]].push_back(label[a], label[b]);
        }
    }

    components(const input_graph& g) : components(g, unaryint<bool>(false)) {}

    components(const input_graph& g, const std::vector<int> reserve)
            : components(g, unaryint<std::vector<int>>(reserve)) {}

    //! Get the set of nodes in a component
    const std::vector<int>& nodes(int c) const { return component[c]; }

    //! Get the number of connected components in the graph
    size_t size() const { return component_g.size(); }

    //! returns the number of reserved nodes in a component
    size_t num_reserved(int c) const { return _num_reserved[c]; }

    //! Get the size (in nodes) of a component
    size_t size(int c) const { return component_g[c].num_nodes(); }

    //! Get a const reference to the graph object of a component
    const input_graph& component_graph(int c) const { return component_g[c]; }

    //! Construct a neighborhood list for component c, with reserved nodes as sources
    std::vector<std::vector<int>> component_neighbors(int c) const {
        return component_g[c].get_neighbors_sources(static_cast<int>(size(c)) - static_cast<int>(num_reserved(c)));
    }

    //! translate nodes from the input graph, to their labels in component c
    template <typename T>
    bool into_component(const int c, T& nodes_in, std::vector<int>& nodes_out) const {
        for (auto& x : nodes_in) {
            try {
                if (index.at(x) != c) return false;
            } catch (std::out_of_range& /*e*/) {
                return false;
            }
            nodes_out.push_back(label[x]);
        }
        return true;
    }

    //! translate nodes from labels in component c, back to their original input labels
    template <typename T>
    void from_component(const int c, T& nodes_in, std::vector<int>& nodes_out) const {
        auto& comp = component[c];
        for (auto& x : nodes_in) {
            nodes_out.push_back(comp[x]);
        }
    }

  private:
    int __init_find(int x) {
        // NEVER CALL AFTER INITIALIZATION
        std::vector<int>& parent = index;
        if (parent[x] != x) parent[x] = __init_find(parent[x]);
        return parent[x];
    }

    void __init_union(int x, int y) {
        // NEVER CALL AFTER INITIALIZATION
        std::vector<int>& parent = index;
        std::vector<int>& rank = label;
        int xroot = __init_find(x);
        int yroot = __init_find(y);
        if (xroot == yroot)
            return;
        else if (rank[xroot] < rank[yroot])
            parent[xroot] = yroot;
        else if (rank[xroot] > rank[yroot])
            parent[yroot] = xroot;
        else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }

    std::vector<int> index;  // NOTE: dual-purpose -- parent for the union/find; index-of-component later
    std::vector<int> label;  // NOTE: dual-purpose -- rank for the union/find; label-in-component later
    std::vector<int> _num_reserved;
    std::vector<std::vector<int>> component;
    std::vector<input_graph> component_g;
};
}  // namespace graph
