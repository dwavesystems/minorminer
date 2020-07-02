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
#include "cell_cache.hpp"

namespace busclique {

template<typename topo_spec>
class bundle_cache {
//  public:
    const cell_cache<topo_spec> &cells;
//  private:
    const size_t linestride[2];
    const size_t orthstride;
    uint8_t *line_mask;
    //prevent double-frees by forbidding moving & copying
    bundle_cache(const bundle_cache&) = delete;
    bundle_cache(bundle_cache &&) = delete;

  public:
    ~bundle_cache() {
        if (line_mask != nullptr) {
            delete [] line_mask;
            line_mask = nullptr;
        }
    }
    bundle_cache(const cell_cache<topo_spec> &c) :
                 cells(c),
                 linestride{binom(c.topo.dim[0]), binom(c.topo.dim[1])},
                 orthstride(c.topo.dim[1]*linestride[0]),
                 line_mask(new uint8_t[orthstride + c.topo.dim[0]*linestride[1]]{}) {
        compute_line_masks();
    }

    size_t score(size_t yc, size_t xc, size_t y0, size_t y1, size_t x0, size_t x1) const {
        return min(get_line_score(0, xc, y0, y1), get_line_score(1, yc, x0, x1));
    }

    void inflate(size_t yc, size_t xc, size_t y0, size_t y1, size_t x0, size_t x1,
                 vector<vector<size_t>> &emb) const {
        uint8_t k0 = get_line_mask(0, xc, y0, y1);
        uint8_t k1 = get_line_mask(1, yc, x0, x1);
        while (k0 && k1) {
            emb.emplace_back(0);
            vector<size_t> &chain = emb.back();
            cells.topo.construct_line(0, xc, y0, y1, first_bit[k0], chain);
            cells.topo.construct_line(1, yc, x0, x1, first_bit[k1], chain);
            k0 ^= mask_bit[first_bit[k0]];
            k1 ^= mask_bit[first_bit[k1]];
        }
    }

    void inflate(size_t y0, size_t y1, size_t x0, size_t x1,
                 vector<vector<size_t>> &emb) const {
        inflate(0, y0, y1, x0, x1, emb);
        inflate(1, y0, y1, x0, x1, emb);
    }

    void inflate(size_t u, size_t y0, size_t y1, size_t x0, size_t x1,
                 vector<vector<size_t>> &emb) const {
        size_t w0, w1, z0, z1;
        if(u) {
            w0 = y0; w1 = y1; z0 = x0; z1 = x1;
        } else {
            w0 = x0; w1 = x1; z0 = y0; z1 = y1;
        }
        for(size_t w = w0; w <= w1; w++) {
            uint8_t k = get_line_mask(u, w, z0, z1);
            while(k) {
                emb.emplace_back(0);
                vector<size_t> &chain = emb.back();
                cells.topo.construct_line(u, w, z0, z1, first_bit[k], chain);
                k ^= mask_bit[first_bit[k]];
            }
        }
    }

    size_t length(size_t yc, size_t xc, size_t y0, size_t y1, size_t x0, size_t x1) const {
        return cells.topo.line_length(0, xc, y0, y1) + cells.topo.line_length(1, yc, x0, x1);
    }

    inline uint8_t get_line_score(size_t u, size_t w, size_t z0, size_t z1) const {
        return popcount[get_line_mask(u, w, z0, z1)];
    }

  private:
    inline uint8_t get_line_mask(size_t u, size_t w, size_t z0, size_t z1) const {
        minorminer_assert(u < 2);
        minorminer_assert(w < cells.topo.dim[1-u]);
        minorminer_assert(z0 <= z1);
        minorminer_assert(z1 < cells.topo.dim[u]);
        return line_mask[u*orthstride + w*linestride[u] + binom(z1) + z0];
    }

    void compute_line_masks() {
        for(size_t u = 0; u < 2; u++) {
            for (size_t w = 0; w < cells.topo.dim[1-u]; w++) {
                for (size_t z = 0; z < cells.topo.dim[u]; z++) {
                    uint8_t *t = line_mask + u*orthstride + w*linestride[u] + binom(z);
                    uint8_t m = t[z] = cells.qmask(u, w, z);
                    for(size_t z0 = z; z0--;)
                        m = t[z0] = m & cells.emask(u, w, z0+1);
                }
            }
        }
    }
};

}
