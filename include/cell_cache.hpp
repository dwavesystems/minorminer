#pragma once
#include "util.hpp"

namespace busclique {

template<typename T> class cell_cache;

template<>
class cell_cache<pegasus_spec> {
    bool borrow;
  public:
    const pegasus_spec topo;
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

    cell_cache(const pegasus_spec p, const vector<size_t> &nodes,
               const vector<pair<size_t, size_t>> &edges) :
               borrow(false), topo(p),
               nodemask(new uint8_t[p.num_cells()]{}),
               edgemask(new uint8_t[p.num_cells()]{}) {
        topo.process_nodes(nodemask, edgemask, nodes);
        topo.process_edges(edgemask, edges);
    }

    cell_cache(const pegasus_spec p, uint8_t *nm, uint8_t *em) :
               borrow(true), topo(p), nodemask(nm), edgemask(em) {}

  public:
    uint8_t qmask(size_t u, size_t w, size_t z) const {
        return nodemask[topo.cell_addr(u, w, z)];
    }
    uint8_t emask(size_t u, size_t w, size_t z) const {
        return edgemask[topo.cell_addr(u, w, z)];
    }

    void construct_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                        vector<size_t> &chain) const {
        size_t qk = (2*w + k)%12;
        size_t qw = (2*w + k)/12;
        size_t qz0 = (z0*2 - 2*topo.offsets[u][qk/2])/12;
        size_t qz1 = (z1*2 - 2*topo.offsets[u][qk/2])/12;
        for(size_t qz = qz0; qz <= qz1; qz++)
            chain.push_back(pegasus_linear(topo.pdim, u, qw, qk, qz));
    }

    size_t line_length(size_t u, size_t w, size_t z0, size_t z1) const {
        size_t qk = (2*w)%12;
        if(z0 < topo.offsets[u][w%6]) return std::numeric_limits<size_t>::max();
        size_t qz0 = (z0 - topo.offsets[u][qk/2])/6;
        size_t qz1 = (z1 - topo.offsets[u][qk/2])/6;
        if(qz1 > topo.pdim-1) return std::numeric_limits<size_t>::max();
        return qz1 - qz0 + 1;
    }

};

template<>
class cell_cache<chimera_spec> {
    const bool borrow;
  public:
    const chimera_spec topo;
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

    cell_cache(const chimera_spec c, const vector<size_t> &nodes,
               const vector<pair<size_t, size_t>> &edges) : borrow(false),
               topo(c),
               nodemask(new uint8_t[c.num_cells()]{}),
               edgemask(new uint8_t[c.num_cells()]{}) {
        for(auto &q: nodes) add_node(q);
        for(auto &e: edges) add_edge (e.first, e.second);
        for(auto &q: nodes) {
            size_t u, w, z;
            topo.linemajor(q, u, w, z);
        }
    }

    cell_cache(const chimera_spec c, uint8_t *nm, uint8_t *em) :
               borrow(true), topo(c),
               nodemask(nm), edgemask(em) {}

  private:
    void add_node(size_t q) {
        size_t u, w, z, k;
        topo.linemajor(q, u, w, z, k);
        nodemask[topo.cell_addr(u, w, z)] |= mask_bit[k];
    }
    
    void add_edge(size_t p, size_t q) {
        size_t pu, pw, pz, pk;
        size_t qu, qw, qz, qk;
        topo.linemajor(p, pu, pw, pz, pk);
        topo.linemajor(q, qu, qw, qz, qk);
        if(pu == qu && pw == qw && pk == qk) {
            if(pz == qz+1)
                edgemask[topo.cell_addr(pu, pw, pz)] |= mask_bit[pk];
            if(qz == pz+1)
                edgemask[topo.cell_addr(qu, qw, qz)] |= mask_bit[qk];
        }
    }

  public:
    uint8_t qmask(size_t u, size_t w, size_t z) const {
        return nodemask[topo.cell_addr(u, w, z)];
    }
    uint8_t emask(size_t u, size_t w, size_t z) const {
        return edgemask[topo.cell_addr(u, w, z)];
    }

    void construct_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                        vector<size_t> &chain) const {
        size_t p[2]; 
        size_t &z = p[u];
        p[1-u] = w;
        for(z = z0; z <= z1; z++)
            chain.push_back(topo.chimera_linear(p[0], p[1], u, k));
    }

    size_t line_length(size_t u, size_t w, size_t z0, size_t z1) const {
        return z1 - z0 + 1;
    }

};

}
