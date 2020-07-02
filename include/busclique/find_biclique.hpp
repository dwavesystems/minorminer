// Copyright 2020 D-Wave Systems Inc.
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

#include "util.hpp"
#include "cell_cache.hpp"
#include "bundle_cache.hpp"
#include "biclique_cache.hpp"
#include "topo_cache.hpp"
#include<unordered_map>


namespace busclique {


class craphash {
  public:
    size_t operator()(const pair<size_t, size_t> x) const { return (x.first << 16) + x.second; }
};

template<typename value_t>
using biclique_result_cache = std::unordered_map<pair<size_t, size_t>,
                                                 value_t,
                                                 craphash>;

template<typename topo_spec>
void best_bicliques(const topo_spec &topo,
                    const vector<size_t> &nodes,
                    const vector<pair<size_t, size_t>> &edges,
                    vector<pair<pair<size_t, size_t>, vector<vector<size_t>>>> &embs) {
    topo_cache<topo_spec> topology(topo, nodes, edges);
    best_bicliques(topology, embs);
}

template<typename topo_spec>
void best_bicliques(topo_cache<topo_spec> &topology,
                    vector<pair<pair<size_t, size_t>, vector<vector<size_t>>>> &embs) {
    embs.clear();
    topology.reset();
    biclique_result_cache<size_t> chainlength;
    biclique_result_cache<vector<vector<size_t>>> emb_cache;
    do {
        bundle_cache<topo_spec> bundles(topology.cells);
        biclique_cache<topo_spec> bicliques(topology.cells, bundles);
        biclique_yield_cache<topo_spec> bcc(topology.cells, bundles, bicliques);
        for(auto z: bcc) {
            size_t s0 = std::get<0>(z);
            size_t s1 = std::get<1>(z);
            size_t cl = std::get<2>(z);
            auto w = chainlength.find(std::make_pair(s0, s1));
            if(w == chainlength.end() || (*w).second > cl)
                emb_cache[std::make_pair(s0, s1)] = std::get<3>(z);
        }
    } while(topology.next());
    for(auto &z: emb_cache)
        embs.push_back(z);
}

}
