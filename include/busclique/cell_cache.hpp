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
    uint8_t qmask(size_t u, size_t w, size_t z) const {
        return nodemask[topo.cell_addr(u, w, z)];
    }
    uint8_t emask(size_t u, size_t w, size_t z) const {
        return edgemask[topo.cell_addr(u, w, z)];
    }

};

}
