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
#include<limits>
#include<utility>
#include<tuple>
#include<vector>
#include<cstdint>
#include<algorithm>
#include<iostream>
#include<string.h>
#include<set>
#include<map>
#include "../debug.hpp"

namespace busclique {
using std::numeric_limits;
using std::vector;
using std::pair;
using std::min;
using std::max;

enum corner : size_t { 
    NW = 1,
    NE = 2,
    SW = 4,
    SE = 8,
    NWskip = 16,
    NEskip = 32,
    SWskip = 64,
    SEskip = 128,
    skipmask = 255-15,
    shift = 8,
    mask = 255,
    none = 0
};

inline size_t binom(size_t x) { return (x*x+x)/2; }

const uint8_t popcount[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 
1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4,
5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

const uint8_t first_bit[256] = {0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 
4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0,
1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0,
1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2,
0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};

const uint8_t mask_bit[8] = {1, 2, 4, 8, 16, 32, 64, 128};
const uint16_t mask_subsets[8] = {1, 2, 4, 8, 16, 32, 64, 128};

const std::set<size_t> _emptyset;

class ignore_badmask {};
class populate_badmask {};
class topo_spec_base {
  public:
    const size_t dim[2];
    const size_t shore;
    topo_spec_base(size_t d0, size_t d1, size_t s) : dim{d0, d1}, shore(s) {}
    template<typename T>
    topo_spec_base(T d[2], size_t s) : topo_spec_base(d[0], d[1], s) {}
    inline size_t cell_addr(size_t u, size_t y, size_t x) const {
        return x + dim[1]*(y + dim[0]*u);
    }
    inline size_t num_cells() const {
        return dim[0]*dim[1]*2;
    }

    inline void linemajor(size_t q, size_t &u, size_t &w, size_t &z, size_t &k) const {
        k = q % shore;
        linemajor(q, u, w, z);
    }
    inline void linemajor(size_t q, size_t &u, size_t &w, size_t &z) const {
        size_t p[2];
                   q/= shore;
        u = q % 2; q /= 2;
        p[0] = q % dim[1];
        p[1] = q / dim[1];
        w = p[u];
        z = p[1-u];
    }

    inline size_t chimera_linear(size_t y, size_t x, size_t u, size_t k) const {
        return k + shore*(u + 2*(x + dim[1]*y));
    }

    inline void chimera_coordinates(size_t q,
                                    size_t &y, size_t &x, size_t &u, size_t &k) const {
        k = q % shore;  q /= shore;
        u = q % 2;      q /= 2;
        x = q % dim[1]; y  = q/dim[1];
    }
};

class pegasus_spec_base : public topo_spec_base {
    using super = topo_spec_base;
  public:
    const size_t pdim;
    const uint8_t offsets[2][6];
    template<typename T>
    pegasus_spec_base(size_t d, T voff, T hoff) : super(6*d, 6*d, 2), pdim(d),
        offsets{{voff[0], voff[1], voff[2], voff[3], voff[4], voff[5]}, 
                {hoff[0], hoff[1], hoff[2], hoff[3], hoff[4], hoff[5]}} {}
  protected:
    template<typename badmask_behavior>
    inline void process_edges(uint8_t *edgemask, uint8_t *badmask, 
                              const vector<pair<size_t, size_t>> &edges,
                              badmask_behavior) const {
        for(auto &e: edges) {
            size_t p = e.first, q = e.second;

            //ensure some normalization: p < q
            if(q < p) std::swap(p, q);
            size_t qu, qw, qk, qz;
            size_t pu, pw, pk, pz;
            pegasus_coordinates(q, qu, qw, qk, qz);
            pegasus_coordinates(p, pu, pw, pk, pz);
            if(pu == qu) {
                if(pw == qw && pk == qk && qz == pz + 1) {
                    //p < q; we place the edgemask on the larger qubit z
                    //and don't futz with the "pz == qz + 1" case
                    size_t z = (qz*12 + 2*offsets[qu][qk/2])/2;
                    size_t w = (qw*12 + qk)/2;
                    edgemask[super::cell_addr(pu, w, z)] |= mask_bit[qk&1];
                } else if (pw == qw && pk == (qk^1) && qz == pz) {
                } else { std::cout << "urp" << std::endl; throw 10; }
            } else {
                if(std::is_same<badmask_behavior, populate_badmask>::value) {
                    //p < q, so pu = 0 and qu = 1
                    size_t y = (qw*12 + qk) / 2;
                    size_t x = (pw*12 + pk) / 2;
                    badmask[super::chimera_linear(y, x, 0, pk&1)] &= ~mask_bit[qk&1];
                    badmask[super::chimera_linear(y, x, 1, qk&1)] &= ~mask_bit[pk&1];
                }
            }
        }
    }

    inline void first_fragment(size_t q, size_t &u, size_t &w, size_t &k, size_t &z) const {
        size_t qw, qk, qz;
        pegasus_coordinates(q, u, qw, qk, qz);
        z = (qz*12 + 2*offsets[u][qk/2])/2;
        w = (qw*12 + qk)/2;
        k = qk&1;
    }

    template<typename badmask_behavior>
    inline void process_nodes(uint8_t *nodemask, uint8_t *edgemask, uint8_t *badmask,
                             const vector<size_t> &nodes, badmask_behavior) const {
        for(auto &q: nodes) {
            size_t u, w, k, z;
            first_fragment(q, u, w, k, z);
            nodemask[super::cell_addr(u, w, z)] |= mask_bit[k];
            if(std::is_same<badmask_behavior, populate_badmask>::value) {
                if(u) { badmask[super::chimera_linear(w, z, 1, k)] = ~0; }
                else  { badmask[super::chimera_linear(z, w, 0, k)] = ~0; }
            }
            for(size_t i = 1; z++, i < 6; i++) {
                size_t curr = super::cell_addr(u, w, z);
                nodemask[curr] |= mask_bit[k];
                edgemask[curr] |= mask_bit[k];
                if(std::is_same<badmask_behavior, populate_badmask>::value) {
                    if(u) { badmask[super::chimera_linear(w, z, 1, k)] = ~0; }
                    else  { badmask[super::chimera_linear(z, w, 0, k)] = ~0; }
                }
            }
        }
    }

    template<typename cell_cache>
    bool has_qubits(const cell_cache &cells, const vector<vector<size_t>> &emb) const {
        for(auto &chain: emb)
            for(auto &q: chain) {
                size_t u, w, k, z0;
                first_fragment(q, u, w, k, z0);
                for(size_t z = z0; z < z0+6; z++)
                    if((cells.qmask(u, w, z) & mask_bit[k]) == 0)
                        return false;
            }
        return true;
    }

  public:
    inline size_t pegasus_linear(size_t u, size_t w, size_t k, size_t z) const {
        return z + (pdim-1)*(k + 12*(w + pdim*u));
    }

    inline void pegasus_coordinates(size_t q,
                                    size_t &u, size_t &w, size_t &k, size_t &z) const {
        z = q % (pdim-1); q /= pdim-1;
        k = q % 12;       q /= 12;
        w = q % pdim;     u = q/pdim;
    }


    void construct_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                        vector<size_t> &chain) const {
        size_t qk = (2*w + k)%12;
        size_t qw = (2*w + k)/12;
        size_t qz0 = (z0*2 - 2*offsets[u][qk/2])/12;
        size_t qz1 = (z1*2 - 2*offsets[u][qk/2])/12;
        for(size_t qz = qz0; qz <= qz1; qz++)
            chain.push_back(pegasus_linear(u, qw, qk, qz));
    }

    inline size_t line_length(size_t u, size_t w, size_t z0, size_t z1) const {
        size_t offset = offsets[u][w%6];
        size_t qz0 = (z0 + 6 - offset)/6;
        size_t qz1 = (z1 + 12 - offset)/6;
        return qz1 - qz0;
    }

    inline size_t biclique_length(size_t y0, size_t y1, size_t x0, size_t x1) const {
        size_t length = 0;
        size_t ym = std::min(y1, y0+5);
        size_t xm = std::min(x1, x0+5);
        for(size_t x = x0; x <= xm; x++)
            length = max(line_length(0, x, y0, y1), length);
        for(size_t y = y0; y <= ym; y++)
            length = max(line_length(1, y, x0, x1), length);
        return length;
    }
};



class chimera_spec_base : public topo_spec_base {
    using super = topo_spec_base;
  public:
    template<typename ...Args>
    chimera_spec_base(Args ...args) : super(args...) {}
  protected:
    template<typename badmask_behavior>
    inline void process_edges(uint8_t *edgemask, uint8_t *badmask, 
                              const vector<pair<size_t, size_t>> &edges,
                              badmask_behavior) const {
        for(auto &e: edges) {
            size_t p = e.first, q = e.second;
            size_t pu, pw, pz, pk;
            size_t qu, qw, qz, qk;
            super::linemajor(p, pu, pw, pz, pk);
            super::linemajor(q, qu, qw, qz, qk);
            if(pu == qu && pw == qw && pk == qk) {
                if(pz == qz+1)
                    edgemask[super::cell_addr(pu, pw, pz)] |= mask_bit[pk];
                if(qz == pz+1)
                    edgemask[super::cell_addr(qu, qw, qz)] |= mask_bit[qk];
            } else if (std::is_same<badmask_behavior, populate_badmask>::value &&
                       pu != qu && pw == qz && pz == qw) {
                badmask[p] &= ~mask_bit[qk];
                badmask[q] &= ~mask_bit[pk];
            }
        }
    }
    template<typename badmask_behavior>
    inline void process_nodes(uint8_t *nodemask, uint8_t *, uint8_t *badmask,
                             const vector<size_t> &nodes, badmask_behavior) const {
        for(auto &q: nodes) {
            //add the node q by updating nodemask on the k-th bit of q's line,
            //and initializing badmask[k] to all-ones
            size_t u, w, z, k;
            super::linemajor(q, u, w, z, k);
            if(std::is_same<badmask_behavior, populate_badmask>::value)
                badmask[q] = ~0;
            nodemask[super::cell_addr(u, w, z)] |= mask_bit[k];
        }
    }

  public:
    void construct_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                        vector<size_t> &chain) const {
        size_t p[2]; 
        size_t &z = p[u];
        p[1-u] = w;
        for(z = z0; z <= z1; z++)
            chain.push_back(super::chimera_linear(p[0], p[1], u, k));
    }

    inline size_t line_length(size_t, size_t, size_t z0, size_t z1) const {
        return z1 - z0 + 1;
    }

    inline size_t biclique_length(size_t y0, size_t y1, size_t x0, size_t x1) const {
        return max(y1-y0, x1-x0) + 1;
    }

};

template<typename topo_spec>
class topo_spec_cellmask : public topo_spec {
    using super = topo_spec;
  public:
    template<typename ...Args>
    topo_spec_cellmask(Args ...args) : super(args...) {}
    inline void process_edges(uint8_t *edgemask, 
                              const vector<pair<size_t, size_t>> &edges) const {
        super::process_edges(edgemask, nullptr, edges, ignore_badmask{});
    }
    inline void process_edges(uint8_t *edgemask, uint8_t *badmask,
                              const vector<pair<size_t, size_t>> &edges) const {
        super::process_edges(edgemask, badmask, edges, populate_badmask{});
    }
    inline void process_nodes(uint8_t *nodemask, uint8_t *edgemask, 
                              const vector<size_t> &nodes) const {
        super::process_nodes(nodemask, edgemask, nullptr, nodes, ignore_badmask{});
    }
    inline void process_nodes(uint8_t *nodemask, uint8_t *edgemask, uint8_t *badmask,
                              const vector<size_t> &nodes) const {
        super::process_nodes(nodemask, edgemask, badmask, nodes, populate_badmask{});
    }
    inline void finish_badmask(uint8_t *nodemask, uint8_t *badmask) const {
        for(size_t y = 0, q = 0; y < super::dim[0]; y++)
            for(size_t x = 0; x < super::dim[1]; x++) {
                for(size_t k = 0; k < super::shore; k++, q++)
                    badmask[q] &= nodemask[super::cell_addr(1, y, x)];
                for(size_t k = 0; k < super::shore; k++, q++)
                    badmask[q] &= nodemask[super::cell_addr(0, x, y)];
            }
    }
};

using chimera_spec = topo_spec_cellmask<chimera_spec_base>;
using pegasus_spec = topo_spec_cellmask<pegasus_spec_base>;

}

