#pragma once
#include<limits>
#include<utility>
#include<tuple>
#include<vector>
#include<cstdint>
#include<algorithm>
//#include<assert.h>
#include<iostream>
#include<string.h>

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
    shift = 4,
    mask = 15,
    none = 0
};
using corner::NW;
using corner::NE;
using corner::SW;
using corner::SE;

void Assert(bool thing) {
    if(!thing) throw std::exception();
}

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

inline void pegasus_coordinates(size_t dim, size_t q, 
                                size_t &u, size_t &w, size_t &k, size_t &z) {
    z = q % (dim-1); q /= dim-1;
    k = q % 12;      q /= 12;
    w = q % dim;     u = q/dim;
}

inline void chimera_coordinates(size_t xdim, size_t shore, size_t q,
                                size_t &y, size_t &x, size_t &u, size_t &k) {
    k = q % shore;  q /= shore;
    u = q % 2;      q /= 2;
    x = q % xdim;   y  = q/xdim;
}
inline size_t pegasus_linear(size_t dim, size_t u, size_t w, size_t k, size_t z) {
    return z + (dim-1)*(k + 12*(w + dim*u));
}

inline size_t chimera_to_linear(size_t xdim, size_t shore, size_t y, size_t x, size_t u, size_t k) {
    return k + shore*(u + 2*(x + xdim*y));
}

inline size_t block_addr(size_t ydim, size_t xdim, size_t u, size_t y, size_t x) {
    return x + xdim*(y + ydim*u);
}

inline size_t block_addr(size_t dim, size_t u, size_t y, size_t x) {
    return block_addr(dim, dim, u, y, x);
}

struct pegasus_spec {
    const size_t dim;
    const uint8_t offsets[2][6];
    template<typename T>
    pegasus_spec(size_t d, T voff, T hoff) :
        dim(d),
        offsets{{voff[0], voff[1], voff[2], voff[3], voff[4], voff[5]}, 
                {hoff[0], hoff[1], hoff[2], hoff[3], hoff[4], hoff[5]}} {}
};

struct chimera_spec {
    const size_t dim[2];
    const uint8_t shore;
    template<typename T>
    chimera_spec(T d, uint8_t s) : dim{d[0], d[1]}, shore(s) {}
    chimera_spec(size_t d0, size_t d1, uint8_t s) : dim{d0, d1}, shore(s) {}
};

}

