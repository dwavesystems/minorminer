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
#include "../fastrng.hpp"

namespace busclique {
using std::numeric_limits;
using std::vector;
using std::pair;
using std::min;
using std::max;
using fastrng::fastrng;

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



class serialize_size_tag {};
class serialize_write_tag {} ;


template<typename T>
class fat_pointer{
    T *ptr;
  public:
    size_t size;
    fat_pointer(size_t size, size_t) : ptr(new T[size]{}), size(size) {}
    fat_pointer(size_t size) : ptr(new T[size]), size(size) {}
    ~fat_pointer() { delete[] ptr; ptr = nullptr; }
    T &operator[](size_t i) { return ptr[i]; }
    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};


//! _serial_helper both computes the size of a field of an object being 
//! serialized, and also writes it into an output buffer, advancing its buffer.
//! The construction provides a single source of truth for the serialization of
//! an object field or data pointer.
template<typename T>
inline size_t _serial_helper(serialize_size_tag, uint8_t *, const fat_pointer<T> &value) {
    return sizeof(T[value.size]);
}

template<typename T>
inline size_t _serial_helper(serialize_size_tag, uint8_t *, const T &value) {
    return sizeof(T);
}

template<typename T>
const T* _serial_addr(const fat_pointer<T> &value) {
    return value;
}

template<typename T> 
const T* _serial_addr(const T &value) {
    return &value;
}


template<typename serialize_tag, typename T>
inline size_t _serial_helper(serialize_tag, uint8_t *output, const T &value) {
    size_t size = _serial_helper(serialize_size_tag{}, output, value);
    if (std::is_same<serialize_tag, serialize_write_tag>::value)
        memcpy(output, _serial_addr(value), size);
    return size;
}

//! _serialize computes the size of a sequence of fields associated with an
//! object being serialized, and also writes those fields into an output buffer
//! advancing the buffer by the corresponding amount.  This provides a single
//! source of truth for the serialization of a class.
template<typename serialize_tag, typename ...>
inline size_t _serialize(serialize_tag, uint8_t *) {
        return 0;
}

template<typename serialize_tag, typename T, typename ...Args>
inline size_t _serialize(serialize_tag, uint8_t *output, const T &value, Args &...args) {
    size_t offset = _serial_helper(serialize_tag{}, output, value);
    return offset + _serialize(serialize_tag{}, output + offset, args...);
}

class ignore_badmask {};
class populate_badmask {};
class topo_spec_base {
  public:
    const size_t dim[2];
    const size_t shore;
    const uint64_t seed;
    
    topo_spec_base(size_t d0, size_t d1, size_t s, uint64_t e) :
        dim{d0, d1}, shore(s), seed(e) {}

    topo_spec_base(size_t d0, size_t d1, size_t s, uint32_t e) :
        topo_spec_base(d0, d1, s, fastrng::amplify_seed(e)) {}

    template<typename serialize_tag, typename ...Args>
    size_t serialize(serialize_tag, uint8_t *output, Args &...args) const {
        return _serialize(serialize_tag{}, output, dim, shore, seed, args...);
    }

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
                   q /= shore;
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
    template<typename T, typename S>
    pegasus_spec_base(size_t d, T voff, T hoff, S seed) :
        super(6*d, 6*d, 2, seed), pdim(d),
        offsets{{voff[0], voff[1], voff[2], voff[3], voff[4], voff[5]}, 
                {hoff[0], hoff[1], hoff[2], hoff[3], hoff[4], hoff[5]}} {}
    static constexpr size_t clique_number = 4;

    template<typename serialize_tag, typename ...Args>
    size_t serialize(serialize_tag, uint8_t *output, Args &...args) const {
        return _serialize(serialize_tag{}, output, offsets, args...);
    }

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
    static constexpr size_t clique_number = 2;
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


class zephyr_spec_base : public topo_spec_base {
    //Note, this differs from pretty much every treatment of zephyr coordinates,
    //though the difference is minor.  Elsewhere, we predominantly use 5-tuple 
    //coordinates (u, w, k, j, z); we unify the k and j indices to preserve the
    //view of Zephyr as a minor of Chimera with K_{8, 8} unit tiles.
    
    //It may be useful to replace the `k` coordinate in these functions with the
    //name `kj` to clarify that k_here = 2*k_elsewhere+j.  Only, do that in your
    //head.

    using super = topo_spec_base;
  public:
    const size_t zdim;
    const size_t zshore;
    template<typename S>
    zephyr_spec_base(size_t d, size_t t, S seed) :
        super(2*d+1, 2*d+1, 2*t, seed), zdim(d), zshore(t) {}
    static constexpr size_t clique_number = 4;

  private:
    inline void first_fragment(size_t q,
                               size_t &u, size_t &w, size_t &k, size_t &z) const {
        z = q % zdim;          q /= zdim;
        k = q % super::shore;  q /= super::shore;
        w = q % super::dim[0]; u = q/super::dim[0];
        z+= z + (k&1);
    }

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
            zephyr_coordinates(q, qu, qw, qk, qz);
            zephyr_coordinates(p, pu, pw, pk, pz);
            if(pu == qu) {
                if(pw == qw && pk == qk && qz == pz + 1) {
                    //p < q; we place the edgemask on the larger qubit z
                    //and don't futz with the "pz == qz + 1" case
                    size_t fz = 2*qz + (qk&1);
                    edgemask[super::cell_addr(qu, qw, fz)] |= mask_bit[qk];
                } else if (pw == qw && pk == (qk^1) && (qz == pz || qz+1 == pz)) {
                } else { std::cout << "urp" << std::endl; throw 10; }
            } else {
                if(std::is_same<badmask_behavior, populate_badmask>::value) {
                    //p < q, so pu = 0 and qu = 1
                    size_t y = qw;
                    size_t x = pw;
                    badmask[super::chimera_linear(y, x, 0, pk)] &= ~mask_bit[qk];
                    badmask[super::chimera_linear(y, x, 1, qk)] &= ~mask_bit[pk];
                }
            }
        }
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
            //advance to the second fragment
            z++;
            size_t curr = super::cell_addr(u, w, z);
            nodemask[curr] |= mask_bit[k];
            edgemask[curr] |= mask_bit[k];
            if(std::is_same<badmask_behavior, populate_badmask>::value) {
                if(u) { badmask[super::chimera_linear(w, z, 1, k)] = ~0; }
                else  { badmask[super::chimera_linear(z, w, 0, k)] = ~0; }
            }
        }
    }

  public:
    inline void zephyr_coordinates(size_t q,
                                   size_t &u, size_t &w, size_t &k, size_t &z) const {
        z = q % zdim;          q /= zdim;
        k = q % super::shore;  q /= super::shore;
        w = q % super::dim[0]; u = q/super::dim[0];
    }

    inline size_t zephyr_linear(size_t u, size_t w, size_t k, size_t z) const {
        return z + zdim*(k + super::shore*(w + super::dim[0]*u));
    }

    void construct_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                        vector<size_t> &chain) const {
        minorminer_assert(z0 >= (k&1));
        minorminer_assert(z1 >= (k&1));
        minorminer_assert(z1 >= z0);
        size_t qz0 = (z0-(k&1))/2;
        size_t qz1 = (z1-(k&1))/2;
        for(size_t qz = qz0; qz <= qz1; qz++)
            chain.push_back(zephyr_linear(u, w, k, qz));
    }

    inline size_t line_length(size_t, size_t, size_t z0, size_t z1, uint8_t k) const {
        minorminer_assert(z0 >= (k&1));
        minorminer_assert(z1 >= (k&1));
        minorminer_assert(z1 >= z0);
        return (z1-(k&1))/2 - (z0-(k&1))/2 + 1;
    }

    inline size_t biclique_length(size_t y0, size_t y1, size_t x0, size_t x1) const {
        minorminer_assert(y1 >= y0);
        minorminer_assert(x1 >= x0);
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
using zephyr_spec = topo_spec_cellmask<zephyr_spec_base>;

}

