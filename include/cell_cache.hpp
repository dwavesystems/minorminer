#pragma once
#include "util.hpp"

namespace busclique {

template<typename T> class cell_cache;

template<>
class cell_cache<pegasus_spec> {
  public:
    const pegasus_spec pegasus;
    const size_t dim[2];
    const size_t shore;
  private:
    uint8_t *nodemask;
    uint8_t *edgemask;
    
  public:
    //prevent double-frees by forbidding moving & copying
    cell_cache(const cell_cache &) = delete;
    cell_cache(cell_cache &&) = delete;
    ~cell_cache() {
        if(nodemask != nullptr) { delete []nodemask; nodemask = nullptr; }
        if(edgemask != nullptr) { delete []edgemask; edgemask = nullptr; }
    }

    cell_cache(const pegasus_spec p, const vector<size_t> &nodes,
                  const vector<pair<size_t, size_t>> &edges) :
                  pegasus(p), dim{6*p.dim, 6*p.dim}, shore(2),
                  nodemask(new uint8_t[2*dim[0]*dim[1]]{}),
                  edgemask(new uint8_t[2*dim[0]*dim[1]]{}) {
        for(auto &q: nodes)
            mark_fragments(q);
        if(pegasus.dim > 2)
            for(auto &e: edges)
                mark_ext(e.first, e.second);
    }

  private:
    void mark_fragments(size_t q) {
        size_t u, qw, qk, qz, w, z;
        pegasus_coordinates(pegasus.dim, q, u, qw, qk, qz);
        z = (qz*12 + 2*pegasus.offsets[u][qk/2])/2;
        w = (qw*12 + qk)/2;
        nodemask[block_addr(dim[0], u, w, z)] |= mask_bit[qk&1];
        for(size_t i = 1; z++, i < 6; i++) {
            size_t curr = block_addr(dim[0], u, w, z);
            nodemask[curr] |= mask_bit[qk&1];
            edgemask[curr] |= mask_bit[qk&1];
        }
    }
    
    void mark_ext(size_t p, size_t q) {
        Assert(pegasus.dim > 2);
        if(p == q+1) { std::swap(p, q); }
        else if (q != p+1) return;
        size_t u, qw, qk, qz, w, z;
        pegasus_coordinates(pegasus.dim, q, u, qw, qk, qz);
        z = (qz*12 + 2*pegasus.offsets[u][qk/2])/2;
        w = (qw*12 + qk)/2;
        edgemask[block_addr(dim[0], u, w, z)] |= mask_bit[qk&1];
    }

  public:
    uint8_t qmask(size_t u, size_t w, size_t z) const {
        return nodemask[block_addr(dim[0], u, w, z)];
    }
    uint8_t emask(size_t u, size_t w, size_t z) const {
        return edgemask[block_addr(dim[0], u, w, z)];
    }

    void construct_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                        vector<size_t> &chain) const {
        size_t qk = (2*w + k)%12;
        size_t qw = (2*w + k)/12;
        size_t qz0 = (z0*2 - 2*pegasus.offsets[u][qk/2])/12;
        size_t qz1 = (z1*2 - 2*pegasus.offsets[u][qk/2])/12;
        for(size_t qz = qz0; qz <= qz1; qz++)
            chain.push_back(pegasus_linear(pegasus.dim, u, qw, qk, qz));
    }

    size_t line_length(size_t u, size_t w, size_t z0, size_t z1) const {
        size_t qk = (2*w)%12;
        if(z0 < pegasus.offsets[u][w%6]) return std::numeric_limits<size_t>::max();
        size_t qz0 = (z0 - pegasus.offsets[u][qk/2])/6;
        size_t qz1 = (z1 - pegasus.offsets[u][qk/2])/6;
        if(qz1 > pegasus.dim-1) return std::numeric_limits<size_t>::max();
        return qz1 - qz0 + 1;
    }

};

template<>
class cell_cache<chimera_spec> {
  public:
    const chimera_spec chimera;
    const size_t dim[2];
    const size_t shore;
  private:
    uint8_t *nodemask;
    uint8_t *edgemask;
    
  public:
    //prevent double-frees by forbidding moving & copying
    cell_cache(const cell_cache &) = delete;
    cell_cache(cell_cache &&) = delete;
    ~cell_cache() {
        if(nodemask != nullptr) { delete []nodemask; nodemask = nullptr; }
        if(edgemask != nullptr) { delete []edgemask; edgemask = nullptr; }
    }

    cell_cache(const chimera_spec c, const vector<size_t> &nodes,
                  const vector<pair<size_t, size_t>> &edges) :
                  chimera(c), dim{c.dim[0], c.dim[1]}, shore(c.shore),
                  nodemask(new uint8_t[2*dim[0]*dim[1]]{}),
                  edgemask(new uint8_t[2*dim[0]*dim[1]]{}) {
        for(auto &q: nodes)
            mark_node(q);
        for(auto &e: edges)
            mark_ext(e.first, e.second);
    }

  private:
    void mark_node(size_t q) {
        size_t y, x, u, k;
        chimera_coordinates(dim[1], shore, q, y, x, u, k);
        if(u) nodemask[block_addr(dim[0], dim[1], u, y, x)] |= mask_bit[k];
        else  nodemask[block_addr(dim[0], dim[1], u, x, y)] |= mask_bit[k];
    }
    
    void mark_ext(size_t p, size_t q) {
        size_t py, px, pu, pk;
        chimera_coordinates(dim[1], shore, p, py, px, pu, pk);
        size_t qy, qx, qu, qk;
        chimera_coordinates(dim[1], shore, q, qy, qx, qu, qk);
        if(pu == qu) {
            if(qu) {
                if(px == qx+1)
                    edgemask[block_addr(dim[0], dim[1], 1, py, px)] |= mask_bit[pk];
                if(qx == px+1)
                    edgemask[block_addr(dim[0], dim[1], 1, qx, qy)] |= mask_bit[qk];
            } else {
                if(py == qy+1)
                    edgemask[block_addr(dim[0], dim[1], 0, px, py)] |= mask_bit[pk];
                if(qy == py+1)
                    edgemask[block_addr(dim[0], dim[1], 0, qy, qx)] |= mask_bit[qk];
            }
        }
    }

  public:
    uint8_t qmask(size_t u, size_t w, size_t z) const {
        return nodemask[block_addr(dim[0], dim[1], u, w, z)];
    }
    uint8_t emask(size_t u, size_t w, size_t z) const {
        return edgemask[block_addr(dim[0], dim[1], u, w, z)];
    }

    void construct_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                        vector<size_t> &chain) const {
        if(u)
            for(size_t z = z0; z <= z1; z++)
                chain.push_back(chimera_to_linear(dim[1], shore, w, z, u, k));
        else
            for(size_t z = z0; z <= z1; z++)
                chain.push_back(chimera_to_linear(dim[1], shore, z, w, u, k));
    }

    size_t line_length(size_t u, size_t w, size_t z0, size_t z1) const {
        return z1 - z0 + 1;
    }

    inline bool checklength(size_t, size_t, size_t, size_t, size_t, size_t, size_t) const {
        return true;
    }
};

}
