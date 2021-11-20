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

#pragma once
#include "util.hpp"

namespace busclique {

template<typename topo_spec>
class cell_cache {
    bool borrow;
  public:
    const topo_spec topo;
  private:
    uint8_t *nodemask;
    uint8_t *edgemask;
    
  public:
    //prevent double-frees by forbidding moving & copying
    cell_cache(const cell_cache &) = delete;
    cell_cache(cell_cache &&) = delete;
    ~cell_cache() {
        if(borrow) return;
        if(nodemask != nullptr) { delete []nodemask; nodemask = nullptr; }
        if(edgemask != nullptr) { delete []edgemask; edgemask = nullptr; }
    }

    cell_cache(const topo_spec p, const vector<size_t> &nodes,
               const vector<pair<size_t, size_t>> &edges) :
               borrow(false), topo(p),
               nodemask(new uint8_t[p.num_cells()]{}),
               edgemask(new uint8_t[p.num_cells()]{}) {
        topo.process_nodes(nodemask, edgemask, nodes);
        topo.process_edges(edgemask, edges);
    }

    cell_cache(const topo_spec p, uint8_t *nm, uint8_t *em) :
               borrow(true), topo(p), nodemask(nm), edgemask(em) {}

  public:
    uint8_t qmask(bool u, size_w w, size_z z) const {
        return nodemask[topo.cell_index(u, w, z)];
    }
    uint8_t emask(bool u, size_w w, size_z z) const {
        return edgemask[topo.cell_index(u, w, z)];
    }

    uint8_t score(size_y y, size_x x) const {
        return min(popcount[qmask(0, vert(x), vert(y))],
                   popcount[qmask(1, horz(y), horz(x))]);
    }

    void inflate(size_y y, size_x x, vector<vector<size_t>> &emb) const {
        uint8_t k0 = qmask(0, vert(x), vert(y));
        uint8_t k1 = qmask(1, horz(y), horz(x));
        while (k0 && k1) {
            emb.emplace_back(0);
            vector<size_t> &chain = emb.back();
            topo.construct_line(0, vert(x), vert(y), vert(y), first_bit[k0], chain);
            topo.construct_line(1, horz(y), horz(x), horz(x), first_bit[k1], chain);
        }
        k0 ^= mask_bit[first_bit[k0]];
        k1 ^= mask_bit[first_bit[k1]];
    }

};

}
