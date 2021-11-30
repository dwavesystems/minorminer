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
#include "coordinate_types.hpp"

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
    const size_t size;
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
    return sizeof(T)*value.size;
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
inline size_t _serialize(serialize_tag, uint8_t *output, const T &value, const Args &...args) {
    size_t offset = _serial_helper(serialize_tag{}, output, value);
    return offset + _serialize(serialize_tag{}, output + offset, args...);
}

class ignore_badmask {};
class populate_badmask {};
class topo_spec_base {
  public:
    const size_y dim_y;
    const size_x dim_x;
    const size_t shore;
    const uint64_t seed;
    
    topo_spec_base(size_t dim_y, size_t dim_x, size_t s, uint64_t e) :
        dim_y(dim_y), dim_x(dim_x), shore(s), seed(e) {}

    topo_spec_base(size_t dim_y, size_t dim_x, size_t s, uint32_t e) :
        topo_spec_base(dim_y, dim_x, s, fastrng::amplify_seed(e)) {}

    template<typename serialize_tag, typename ...Args>
    size_t serialize(serialize_tag, uint8_t *output, const Args &...args) const {
        return _serialize(serialize_tag{}, output, dim_y, dim_x, shore, seed, args...);
    }

    //! cell_index is used for indexing into the nodemask and edgemask fields
    //! of cell_cache and topo_cache objects.  These are used to store bitmasks
    //! indicating the presence of nodes in the corresponding cell, or external
    //! edges between the corresponding cell and its greater parallel neighbor
    //! (that is, between the cells [y,x,0] and [y+1,x,0] or between the cells
    //! [y,x,1] and [y,x+1,1]).
    inline size_t cell_index(size_y y, size_x x, bool u) const {
        return coordinate_converter::cell_index(y, x, u, dim_y, dim_x);
    }

    //! cell_index is used for indexing into the nodemask and edgemask fields
    //! of cell_cache and topo_cache objects.  These are used to store bitmasks
    //! indicating the presence of nodes in the corresponding cell, or external
    //! edges between the corresponding cell and its greater parallel neighbor
    //! (that is, between the cells [u,w,z] and [u,w,z+1])
    inline size_t cell_index(bool u, size_w w, size_z z) const {
        return coordinate_converter::cell_index(u, w, z, dim_y, dim_x);
    }

    inline size_t num_cells() const {
        return coordinate_converter::product(dim_y, dim_x, 2);
    }

    template<typename shore_t>
    inline size_t chimera_linear(size_y y, size_x x, bool u, shore_t k) const {
        return coordinate_converter::chimera_linear(y, x, u, k, dim_y, dim_x, shore_t(shore));
    }

    template<typename shore_t>
    inline void linear_chimera(size_t q, 
                               size_y &y, size_x &x, bool &u, shore_t &k) const {
        coordinate_converter::linear_chimera(q, y, x, u, k, dim_y, dim_x, shore_t(shore));
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
    size_t serialize(serialize_tag, uint8_t *output, const Args &...args) const {
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
            bool    pu, qu;
            size_w  pw, qw;
            uint8_t pk, qk;
            size_z  pz, qz;
            pegasus_coordinates(q, qu, qw, qk, qz);
            pegasus_coordinates(p, pu, pw, pk, pz);
            if(pu == qu) {
                if(pw == qw && pk == qk && qz == pz + 1_z) {
                    //p < q; we place the edgemask on the larger qubit z
                    //and don't futz with the "pz == qz + 1" case
                    size_z z = qz*6_z + size_z(offsets[qu][qk/2]);
                    size_w w = qw*6_w + size_w(qk/2);
                    edgemask[super::cell_index(pu, w, z)] |= mask_bit[qk&1];
                } else if (pw == qw && pk == (qk^1) && qz == pz) {
                } else { std::cout << "urp" << std::endl; throw 10; }
            } else {
                if(std::is_same<badmask_behavior, populate_badmask>::value) {
                    //p < q, so pu = 0 and qu = 1
                    size_y y = horz(qw)*6_y + size_y(qk/2);
                    size_x x = vert(pw)*6_x + size_x(pk/2);
                    badmask[super::chimera_linear(y, x, 0, pk&1)] &= ~mask_bit[qk&1];
                    badmask[super::chimera_linear(y, x, 1, qk&1)] &= ~mask_bit[pk&1];
                }
            }
        }
    }

    inline void first_fragment(size_t q, bool &u, size_w &w, uint8_t &k, size_z &z) const {
        size_w qw;
        uint8_t qk;
        size_z qz;
        pegasus_coordinates(q, u, qw, qk, qz);
        z = qz*6_z + size_z(offsets[u][qk/2]);
        w = qw*6_w + size_w(qk/2);
        k = qk&1;
    }

    template<typename badmask_behavior>
    inline void process_nodes(uint8_t *nodemask, uint8_t *edgemask, uint8_t *badmask,
                             const vector<size_t> &nodes, badmask_behavior) const {
        for(auto &q: nodes) {
            bool    u;
            size_w  w;
            uint8_t k;
            size_z  z;
            first_fragment(q, u, w, k, z);
            nodemask[super::cell_index(u, w, z)] |= mask_bit[k];
            if(std::is_same<badmask_behavior, populate_badmask>::value) {
                if(u) { badmask[super::chimera_linear(horz(w), horz(z), 1, k)] = ~0; }
                else  { badmask[super::chimera_linear(vert(z), vert(w), 0, k)] = ~0; }
            }
            for(size_t i = 1; i < 6; i++) {
                z++;
                nodemask[super::cell_index(u, w, z)] |= mask_bit[k];
                edgemask[super::cell_index(u, w, z)] |= mask_bit[k];
                if(std::is_same<badmask_behavior, populate_badmask>::value) {
                    if(u) { badmask[super::chimera_linear(horz(w), horz(z), 1, k)] = ~0; }
                    else  { badmask[super::chimera_linear(vert(z), vert(w), 0, k)] = ~0; }
                }
            }
        }
    }

    template<typename cell_cache>
    bool has_qubits(const cell_cache &cells, const vector<vector<size_t>> &emb) const {
        for(auto &chain: emb)
            for(auto &q: chain) {
                bool u;
                size_w w;
                uint8_t k;
                size_z z0;
                first_fragment(q, u, w, k, z0);
                for(size_z z = z0; z < z0+6_z; z++)
                    if((cells.qmask(u, w, z) & mask_bit[k]) == 0)
                        return false;
            }
        return true;
    }

  public:
    inline size_t pegasus_linear(bool u, size_w w, uint8_t k, size_z z) const {
        return coordinate_converter::linemajor_linear(u, w, k, z, pdim, uint8_t(12), pdim-1);

    }

    vector<size_t> fragment_nodes(size_t q) const {
        bool u;
        size_w w;
        uint8_t k;
        size_z z0;
        first_fragment(q, u, w, k, z0);
        vector<size_t> fragments;
        for(size_z z = z0+6_z; z-->z0;) {
            fragments.push_back(
                u?super::chimera_linear(horz(w), horz(z), u, k):
                  super::chimera_linear(vert(z), vert(w), u, k)
            );
        }        
        return fragments;
    }

    inline void pegasus_coordinates(size_t q,
                                    bool &u, size_w &w, uint8_t &k, size_z &z) const {
        coordinate_converter::linear_linemajor(q, u, w, k, z, pdim, uint8_t(12), pdim-1);
    }

    void construct_line(bool u, size_w w, size_z z0, size_z z1, uint8_t k,
                        vector<size_t> &chain) const {
        uint8_t qk = (coordinate_index(w)*2 + k)%12;
        size_w qw = (w*2_w + size_w(k))/12_w;
        size_z qz0 = (z0 - size_z(offsets[u][qk/2]))/6_z;
        size_z qz1 = (z1 - size_z(offsets[u][qk/2]))/6_z;
        for(size_z qz = qz0; qz <= qz1; qz++)
            chain.push_back(pegasus_linear(u, qw, qk, qz));
    }

    inline size_t line_length(bool u, size_w w, size_z z0, size_z z1) const {
        size_z offset = offsets[u][coordinate_index(w)%6];
        size_z qz0 = (z0 + 6_z - offset)/6_z;
        size_z qz1 = (z1 + size_z(12) - offset)/6_z;
        return coordinate_index(qz1 - qz0);
    }

    inline size_t biclique_length(size_y y0, size_y y1, size_x x0, size_x x1) const {
        minorminer_assert(y1 >= y0);
        minorminer_assert(x1 >= x0);
        size_t length = 0;
        size_y ym = std::min(y1, y0+5_y);
        size_x xm = std::min(x1, x0+5_x);
        for(size_x x = x0; x <= xm; x++)
            length = max(line_length(0, vert(x), vert(y0), vert(y1)), length);
        for(size_y y = y0; y <= ym; y++)
            length = max(line_length(1, horz(y), horz(x0), horz(x1)), length);
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
            if(q < p) std::swap(p, q);
            size_y  py, qy;
            size_x  px, qx;
            bool    pu, qu;
            uint8_t pk, qk;
            super::linear_chimera(p, py, px, pu, pk);
            super::linear_chimera(q, qy, qx, qu, qk);
            if(pu == qu) {
                if (pk == qk && py + size_y(1-pu) == qy && px + size_x(pu) == qx)
                    edgemask[super::cell_index(qy, qx, qu)] |= mask_bit[qk];
            } else if (std::is_same<badmask_behavior, populate_badmask>::value &&
                       pu != qu && py == qy && px == qx) {
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
            size_y  y;
            size_x  x;
            bool    u;
            uint8_t k;
            super::linear_chimera(q, y, x, u, k);
            if(std::is_same<badmask_behavior, populate_badmask>::value)
                badmask[q] = ~0;
            nodemask[super::cell_index(y, x, u)] |= mask_bit[k];
        }
    }

  public:    
    void construct_line(bool u, size_w w, size_z z0, size_z z1, uint8_t k,
                        vector<size_t> &chain) const {
        if(u) {
            for(size_x x = horz(z0); x <= horz(z1); x++)
                chain.push_back(super::chimera_linear(horz(w), x, 1, k));
        } else {
            for(size_y y = vert(z0); y <= vert(z1); y++)
                chain.push_back(super::chimera_linear(y, vert(w), 0, k));
        }
    }

    template<typename U, typename T>
    inline size_t line_length(bool, U, T t0, T t1) const {
        return coordinate_index(t1 - t0) + 1;
    }

    inline size_t biclique_length(size_y y0, size_y y1, size_x x0, size_x x1) const {
        return max(coordinate_index(y1-y0), coordinate_index(x1-x0)) + 1;
    }

    vector<size_t> fragment_nodes(size_t q) const {
        return vector<size_t>(1, q);
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
                               bool &u, size_w &w, uint8_t &k, size_z &z) const {
        coordinate_converter::linear_linemajor(q, u, w, k, z, 2*zdim+1, uint8_t(super::shore), zdim);
        z = z*2_z + size_z(k&1);
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
            bool    pu, qu;
            size_w  pw, qw;
            uint8_t pk, qk;
            size_z  pz, qz;
            zephyr_coordinates(q, qu, qw, qk, qz);
            zephyr_coordinates(p, pu, pw, pk, pz);
            if(pu == qu) {
                if(pw == qw && pk == qk && qz == pz + 1_z) {
                    //p < q; we place the edgemask on the larger qubit z
                    //and don't futz with the "pz == qz + 1" case
                    size_z fz = qz*2_z + size_z(qk&1);
                    edgemask[super::cell_index(qu, qw, fz)] |= mask_bit[qk];
                } else if (pw == qw && pk == (qk^1) && (qz == pz || qz+1_z == pz)) {
                } else { std::cout << "urp" << std::endl; throw 10; }
            } else {
                if(std::is_same<badmask_behavior, populate_badmask>::value) {
                    //p < q, so pu = 0 and qu = 1
                    size_y y = horz(qw);
                    size_x x = vert(pw);
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
            bool    u;
            size_w  w;
            uint8_t k;
            size_z  z;
            first_fragment(q, u, w, k, z);
            nodemask[super::cell_index(u, w, z)] |= mask_bit[k];
            if(std::is_same<badmask_behavior, populate_badmask>::value) {
                if(u) { badmask[super::chimera_linear(horz(w), horz(z), 1, k)] = ~0; }
                else  { badmask[super::chimera_linear(vert(z), vert(w), 0, k)] = ~0; }
            }
            //advance to the second fragment
            z++;
            nodemask[super::cell_index(u, w, z)] |= mask_bit[k];
            edgemask[super::cell_index(u, w, z)] |= mask_bit[k];
            if(std::is_same<badmask_behavior, populate_badmask>::value) {
                if(u) { badmask[super::chimera_linear(horz(w), horz(z), 1, k)] = ~0; }
                else  { badmask[super::chimera_linear(vert(z), vert(w), 0, k)] = ~0; }
            }
        }
    }

  public:
    inline void zephyr_coordinates(size_t q,
                                   bool &u, size_w &w, uint8_t &k, size_z &z) const {
        coordinate_converter::linear_linemajor(q, u, w, k, z, 2*zdim+1, uint8_t(super::shore), zdim);
    }
  
    inline size_t zephyr_linear(bool u, size_w w, uint8_t k, size_z z) const {
        return coordinate_converter::linemajor_linear(u, w, k, z, 2*zdim+1, uint8_t(super::shore), zdim);
    }

    void construct_line(bool u, size_w w, size_z z0, size_z z1, uint8_t k,
                        vector<size_t> &chain) const {
        minorminer_assert(z0 >= size_z(k&1));
        minorminer_assert(z1 >= size_z(k&1));
        minorminer_assert(z1 >= z0);
        size_z qz0 = (z0-size_z(k&1))/2_z;
        size_z qz1 = (z1-size_z(k&1))/2_z;
        for(size_z qz = qz0; qz <= qz1; qz++)
            chain.push_back(zephyr_linear(u, w, k, qz));
    }

    inline size_t line_length(bool u, size_w, size_z z0, size_z z1, uint8_t k) const {
        minorminer_assert(z0 >= size_z(k&1));
        minorminer_assert(z1 >= size_z(k&1));
        minorminer_assert(z1 >= z0);
        minorminer_assert(u?(horz(z1)<super::dim_x):
                            (vert(z1)<super::dim_y));
        return coordinate_index(((z1-size_z(k&1))/2_z - (z0-size_z(k&1))/2_z)) + 1;
    }

    inline size_t biclique_length(size_y y0, size_y y1, size_x x0, size_x x1) const {
        minorminer_assert(y1 >= y0);
        minorminer_assert(x1 >= x0);
        // we return a length of zero for the ranges (0, super::dim[0]-1)
        // because there are zero qubits that span the entire range.
        size_t length = 0;
        if (y0 > 0_y)
            length = max(length, line_length(0, 0, vert(y0), vert(y1), 1));
        if (x0 > 0_x)
            length = max(length, line_length(1, 0, horz(x0), horz(x1), 1));
        if (y1 + 1_y < super::dim_y)
            length = max(length, line_length(0, 0, vert(y0), vert(y1), 0));
        if (x1 + 1_x < super::dim_x)
            length = max(length, line_length(1, 0, horz(x0), horz(x1), 0));
        return length;
    }

    vector<size_t> fragment_nodes(size_t q) const {
        bool u;
        size_w w;
        uint8_t k;
        size_z z0;
        first_fragment(q, u, w, k, z0);
        vector<size_t> fragments;
        for(size_z z = z0+2_z; z-->z0;) {
            fragments.push_back(
                u?super::chimera_linear(horz(w), horz(z), u, k):
                  super::chimera_linear(vert(z), vert(w), u, k)
            );
        }
        return fragments;
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
        size_t q = 0;
        for(size_y y = 0; y < super::dim_y; y++)
            for(size_x x = 0; x < super::dim_x; x++) {
                for(uint8_t k = 0; k < super::shore; k++) {
                    minorminer_assert(q == super::chimera_linear(y, x, 0, k));
                    badmask[q++] &= nodemask[super::cell_index(y, x, 1)];
                }
                for(uint8_t k = 0; k < super::shore; k++) {
                    minorminer_assert(q == super::chimera_linear(y, x, 1, k));
                    badmask[q++] &= nodemask[super::cell_index(y, x, 0)];
                }
            }
    }
    inline vector<size_t> fragment_nodes(vector<size_t> &nodes) const {
        vector<size_t> fragments;
        for(auto &q: nodes) {
            auto f = super::fragment_nodes(q);
            fragments.insert(fragments.end(), f.begin(), f.end());
        }
        return fragments;
    }
};

using chimera_spec = topo_spec_cellmask<chimera_spec_base>;
using pegasus_spec = topo_spec_cellmask<pegasus_spec_base>;
using zephyr_spec = topo_spec_cellmask<zephyr_spec_base>;

}

