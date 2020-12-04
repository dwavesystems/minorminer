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
#include <random>
#include <set>
#include <string>
#include <vector>

#include "graph.hpp"
#include "pathfinder.hpp"
#include "util.hpp"

namespace find_embedding {

class parameter_processor {
  public:
    unsigned int num_vars;
    unsigned int num_qubits;

    vector<int> qub_reserved_unscrewed;
    vector<int> var_fixed_unscrewed;
    unsigned int num_reserved;

    graph::components qub_components;
    unsigned int problem_qubits;
    unsigned int problem_reserved;

    unsigned int num_fixed;
    vector<int> unscrew_vars;
    vector<int> screw_vars;

    optional_parameters params;
    vector<vector<int>> var_nbrs;
    vector<vector<int>> qubit_nbrs;
    parameter_processor(graph::input_graph &var_g, graph::input_graph &qubit_g, optional_parameters &params_)
            : num_vars(var_g.num_nodes()),
              num_qubits(qubit_g.num_nodes()),

              qub_reserved_unscrewed(num_qubits, 0),
              var_fixed_unscrewed(num_vars, 0),
              num_reserved(_reserved(params_)),

              qub_components(qubit_g, qub_reserved_unscrewed),
              problem_qubits(qub_components.size(0)),
              problem_reserved(qub_components.num_reserved(0)),

              num_fixed(params_.fixed_chains.size()),
              unscrew_vars(_filter_fixed_vars()),
              screw_vars(_inverse_permutation(unscrew_vars)),

              params(params_, input_chains(params_.fixed_chains), input_chains(params_.initial_chains),
                     input_chains(params_.restrict_chains)),

              var_nbrs(var_g.get_neighbors_sinks(var_fixed_unscrewed, screw_vars)),
              qubit_nbrs(qub_components.component_neighbors(0)) {}

  private:
    inline unsigned int _reserved(optional_parameters &params_) {
        unsigned int r = 0;
        for (auto &vC : params_.fixed_chains) {
            var_fixed_unscrewed[vC.first] = 1;
            for (auto &q : vC.second) {
                if (!qub_reserved_unscrewed[q]) {
                    qub_reserved_unscrewed[q] = 1;
                    r++;
                }
            }
        }
        return r;
    }

    vector<int> _filter_fixed_vars() {
        vector<int> unscrew(num_vars);
        assert(var_fixed_unscrewed.size() == num_vars);
        assert(num_fixed < num_vars);
        for (unsigned int i = 0, front = 0, back = num_vars - num_fixed; i < num_vars; i++) {
            if (var_fixed_unscrewed[i]) {
                unscrew[back++] = i;
            } else {
                unscrew[front++] = i;
            }
        }
        return unscrew;
    }

    vector<int> _inverse_permutation(vector<int> &f) {
        size_t n = f.size();
        vector<int> r(n);
        for (size_t i = n; i--;) {
            r.at(f[i]) = i;
        }
        return r;
    }

  public:
    map<int, vector<int>> input_chains(map<int, vector<int>> &m) {
        map<int, vector<int>> n;
        for (auto &kv : m) {
            if (kv.first < 0 || static_cast<unsigned int>(kv.first) >= num_vars) throw CorruptParametersException();
            auto &ju = *(n.emplace(screw_vars[kv.first], vector<int>{}).first);
            if (!qub_components.into_component(0, kv.second, ju.second)) {
                throw CorruptParametersException();
            }
        }
        return n;
    }

    vector<int> input_vars(vector<int> &V) {
        vector<int> U;
        for (auto &v : V) {
            if (v < 0 || static_cast<unsigned int>(v) >= num_vars) throw CorruptParametersException();
            if (!var_fixed_unscrewed[v]) U.push_back(v);
        }
        return U;
    }
};

template <bool parallel, bool fixed, bool restricted, bool verbose>
class pathfinder_type {
  public:
    typedef typename std::conditional<fixed, fixed_handler_hival, fixed_handler_none>::type fixed_handler_t;
    typedef typename std::conditional<restricted, domain_handler_masked, domain_handler_universe>::type
            domain_handler_t;
    typedef output_handler<verbose> output_handler_t;
    typedef embedding_problem<fixed_handler_t, domain_handler_t, output_handler_t> embedding_problem_t;
    typedef typename std::conditional<parallel, pathfinder_parallel<embedding_problem_t>,
                                      pathfinder_serial<embedding_problem_t>>::type pathfinder_t;
};

class pathfinder_wrapper {
    parameter_processor pp;
    std::unique_ptr<pathfinder_public_interface> pf;

  public:
    pathfinder_wrapper(graph::input_graph &var_g, graph::input_graph &qubit_g, optional_parameters &params_)
            : pp(var_g, qubit_g, params_),
              pf(_pf_parse(pp.params, pp.num_vars - pp.num_fixed, pp.num_fixed, pp.problem_qubits - pp.problem_reserved,
                           pp.problem_reserved, pp.var_nbrs, pp.qubit_nbrs)) {}

    ~pathfinder_wrapper() {}

    void get_chain(int u, vector<int> &output) const {
        pp.qub_components.from_component(0, pf->get_chain(pp.screw_vars[u]), output);
    }

    int heuristicEmbedding() { return pf->heuristicEmbedding(); }

    int num_vars() { return pp.num_vars; }

    void set_initial_chains(map<int, vector<int>> &init) { pf->set_initial_chains(pp.input_chains(init)); }

    void quickPass(vector<int> &varorder, int chainlength_bound, int overlap_bound, bool local_search, bool clear_first,
                   double round_beta) {
        pf->quickPass(pp.input_vars(varorder), chainlength_bound, overlap_bound, local_search, clear_first, round_beta);
    }
    void quickPass(VARORDER varorder, int chainlength_bound, int overlap_bound, bool local_search, bool clear_first,
                   double round_beta) {
        pf->quickPass(varorder, chainlength_bound, overlap_bound, local_search, clear_first, round_beta);
    }

  private:
    template <bool parallel, bool fixed, bool restricted, bool verbose, typename... Args>
    inline std::unique_ptr<pathfinder_public_interface> _pf_parse4(Args &&... args) {
        return std::unique_ptr<pathfinder_public_interface>(static_cast<pathfinder_public_interface *>(
                new (typename pathfinder_type<parallel, fixed, restricted, verbose>::pathfinder_t)(
                        std::forward<Args>(args)...)));
    }

    template <bool parallel, bool fixed, bool restricted, typename... Args>
    inline std::unique_ptr<pathfinder_public_interface> _pf_parse3(Args &&... args) {
        if (pp.params.verbose <= 0)
            return _pf_parse4<parallel, fixed, restricted, false>(std::forward<Args>(args)...);
        else
            return _pf_parse4<parallel, fixed, restricted, true>(std::forward<Args>(args)...);
    }

    template <bool parallel, bool fixed, typename... Args>
    inline std::unique_ptr<pathfinder_public_interface> _pf_parse2(Args &&... args) {
        if (pp.params.restrict_chains.size())
            return _pf_parse3<parallel, fixed, true>(std::forward<Args>(args)...);
        else
            return _pf_parse3<parallel, fixed, false>(std::forward<Args>(args)...);
    }

    template <bool parallel, typename... Args>
    inline std::unique_ptr<pathfinder_public_interface> _pf_parse1(Args &&... args) {
        if (pp.params.fixed_chains.size())
            return _pf_parse2<parallel, true>(std::forward<Args>(args)...);
        else
            return _pf_parse2<parallel, false>(std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline std::unique_ptr<pathfinder_public_interface> _pf_parse(Args &&... args) {
        if (pp.params.threads > 1)
            return _pf_parse1<true>(std::forward<Args>(args)...);
        else
            return _pf_parse1<false>(std::forward<Args>(args)...);
    }
};

//! The main entry function of this library.
//!
//! This method primarily dispatches the proper implementation of the algorithm
//! where some parameters/behaviours have been fixed at compile time.
//!
//! In terms of dispatch, there are three dynamically-selected classes which
//! are combined, each according to a specific optional parameter.
//!   * a domain_handler, described in embedding_problem.hpp, manages
//!     constraints of the form "variable a's chain must be a subset of..."
//!   * a fixed_handler, described in embedding_problem.hpp, manages
//!     contstraints of the form "variable a's chain must be exactly..."
//!   * a pathfinder, described in pathfinder.hpp, which come in two flavors,
//!     serial and parallel
//! The optional parameters themselves can be found in util.hpp.  Respectively,
//! the controlling options for the above are restrict_chains, fixed_chains,
//! and threads.
int findEmbedding(graph::input_graph &var_g, graph::input_graph &qubit_g, optional_parameters &params,
                  vector<vector<int>> &chains) {
    pathfinder_wrapper pf(var_g, qubit_g, params);
    int success = pf.heuristicEmbedding();

    if (params.return_overlap || success) {
        chains.resize(var_g.num_nodes());
        for (size_t u = 0; u < var_g.num_nodes(); u++) {
            pf.get_chain(u, chains[u]);
        }
    } else {
        chains.clear();
    }

    return success;
}
}  // namespace find_embedding
