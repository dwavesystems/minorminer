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
