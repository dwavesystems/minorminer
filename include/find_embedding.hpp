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
  protected:
    int num_vars;
    int num_qubits;

    vector<int> qub_reserved_unscrewed;
    vector<int> var_fixed_unscrewed;
    int num_reserved;

    graph::components qub_components;
    int problem_qubits;
    int problem_reserved;

    int num_fixed;
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
              unscrew_vars(_filter(num_vars, var_fixed_unscrewed)),
              screw_vars(_inverse(unscrew_vars)),

              params(params_, _input_chains(params_.fixed_chains), _input_chains(params_.initial_chains),
                     _input_chains(params_.restrict_chains)),

              var_nbrs(var_g.get_neighbors_sinks(var_fixed_unscrewed, screw_vars)),
              qubit_nbrs(qub_components.component_neighbors(0)) {
        /*
                int t = 0; for(int q = 0; q < num_vars; q++) {
                    //std::cout << " " << var_fixed_unscrewed[q]; t+= var_fixed_unscrewed[q];
                } //std::cout << std::endl; //std::cout << num_vars << "," << num_fixed << "," << t << std::endl;

                t = 0; for(int q = 0; q < num_qubits; q++) {
                    //std::cout << " " << qub_reserved_unscrewed[q]; t+= qub_reserved_unscrewed[q];
                } //std::cout << std::endl; //std::cout << num_qubits << "," << num_reserved << ","<<  t << std::endl;

                t = 0; for(int q = 0; q < num_qubits; q++) {
                    //std::cout << " " << first_component[q]; t += first_component[q];
                } //std::cout << std::endl; //std::cout << num_qubits << "," << connected_qubits << "," << t <<
           std::endl;

                t = 0; for(int q = 0; q < num_qubits; q++) {
                    //std::cout << " " << screw_qubs[q];
                } //std::cout << std::endl;

                t = 0; for(int q = 0; q < num_vars; q++) {
                    //std::cout << " " << screw_vars[q];
                } //std::cout << std::endl;

                //std::cout << qubit_nbrs.size() << "==" << connected_qubits << "?";
                t=0; for(int q=qubit_nbrs.size(); q--;)
                    for(auto &p: qubit_nbrs[q])
                        if(p < 0 or p > connected_qubits) //std::cout << "{" << p << "!" << q << "}";
                        else t++;
                //std::cout << qubit_g.num_edges() << "==" << t << "?" << std::endl;

                //std::cout << var_nbrs.size() << "==" << num_vars << "?";
                t=0; for(int q=var_nbrs.size(); q--;)
                    for(auto &p: var_nbrs[q])
                        if(q > num_vars-num_fixed) //std::cout << "{" << p << "!" << q << "}";
                        else t++;
                //std::cout << var_g.num_edges() << "==" << t << "?" << std::endl;
        */
    }

  private:
    inline int _reserved(optional_parameters &params_) {
        int r = 0;
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

    vector<int> _filter(int n, vector<int> &test) {
        vector<int> r(n);
        // std::cout << ">" << test.size() << "<";
        for (int i = 0, front = 0, back = num_vars - num_reserved; i < n; i++) {
            // std::cout << "(" << i << "," << front << "," << back << ")" << std::endl; std::flush(//std::cout);
            if (test[i]) {
                r[back++] = i;
            } else {
                r[front++] = i;
            }
        }
        for (int i = 0; i < n; i++) {
            // std::cout << "(" << i << "," << r[i] << ")" << std::endl; std::flush(//std::cout);
        }
        return r;
    }

    vector<int> _inverse(vector<int> &f) {
        int n = f.size();
        vector<int> r(n);
        for (int i = n; i--;) {
            r[f[i]] = i;
        }
        return r;
    }

  private:
    map<int, vector<int>> _input_chains(map<int, vector<int>> &m) {
        map<int, vector<int>> n;
        for (auto &kv : m) {
            if (kv.first < 0 || kv.first >= num_vars) throw CorruptParametersException();
            auto &ju = *(n.emplace(screw_vars[kv.first], vector<int>{}).first);
            if (!qub_components.into_component(0, kv.second, ju.second)) {
                throw CorruptParametersException();
            }
        }
        return n;
    }
};

template <typename pathfinder_t>
class pathfinder_wrapper : public parameter_processor, public pathfinder_t {
    using pf = pathfinder_t;
    using pp = parameter_processor;

  public:
    pathfinder_wrapper(graph::input_graph &var_g, graph::input_graph &qubit_g, optional_parameters &params_)
            : parameter_processor(var_g, qubit_g, params_),
              pathfinder_t(pp::params, pp::num_vars - pp::num_fixed, pp::num_fixed,
                           pp::problem_qubits - pp::problem_reserved, pp::num_reserved, pp::var_nbrs, pp::qubit_nbrs) {}

    void get_chain(int u, vector<int> &output) const {
        pp::qub_components.from_component(0, pf::get_chain(pp::screw_vars[u]), output);
    }
};

template <typename pathfinder_t>
int find_embedding_execute(optional_parameters &params, graph::input_graph &var_g, graph::input_graph &qubit_g,
                           vector<vector<int>> &chains) {
    pathfinder_wrapper<pathfinder_t> pf(var_g, qubit_g, params);
    int success = pf.heuristicEmbedding();

    if (params.return_overlap || success) {
        chains.resize(var_g.num_nodes());
        for (int u = 0; u < var_g.num_nodes(); u++) {
            pf.get_chain(u, chains[u]);
        }
    } else {
        chains.clear();
    }

    return success;
}

template <bool parallel, bool fixed, bool restricted>
class pathfinder_type {
  public:
    typedef typename std::conditional<fixed, fixed_handler_hival, fixed_handler_none>::type fixed_handler_t;
    typedef typename std::conditional<restricted, domain_handler_masked, domain_handler_universe>::type
            domain_handler_t;
    typedef embedding_problem<fixed_handler_t, domain_handler_t> embedding_problem_t;
    typedef typename std::conditional<parallel, pathfinder_parallel<embedding_problem_t>,
                                      pathfinder_serial<embedding_problem_t>>::type pathfinder_t;
};

template <bool parallel, bool fixed, bool restricted, typename... Args>
inline int _find_embedding(optional_parameters &params_, Args... args) {
    return find_embedding_execute<typename pathfinder_type<parallel, fixed, restricted>::pathfinder_t>(params_,
                                                                                                       args...);
}

template <bool parallel, bool fixed, typename... Args>
inline int _find_embedding_restricted(optional_parameters &params_, Args... args) {
    if (params_.restrict_chains.size())
        return _find_embedding<parallel, fixed, true>(params_, args...);
    else
        return _find_embedding<parallel, fixed, false>(params_, args...);
}

template <bool parallel, typename... Args>
inline int _find_embedding_fixed(optional_parameters &params_, Args... args) {
    if (params_.fixed_chains.size())
        return _find_embedding_restricted<parallel, true>(params_, args...);
    else
        return _find_embedding_restricted<parallel, false>(params_, args...);
}

template <typename... Args>
inline int _find_embedding_parallel(optional_parameters &params_, Args... args) {
    if (params_.threads > 1)
        return _find_embedding_fixed<true>(params_, args...);
    else
        return _find_embedding_fixed<false>(params_, args...);
}

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
inline int findEmbedding(graph::input_graph &var_g, graph::input_graph &qubit_g, optional_parameters &params_,
                         vector<vector<int>> &chains) {
    return _find_embedding_parallel(params_, var_g, qubit_g, chains);
}
}
