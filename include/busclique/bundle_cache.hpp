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
                 linestride{binom(c.topo.dim_y), binom(c.topo.dim_x)},
                 orthstride(coordinate_index(c.topo.dim_x)*linestride[0]),
                 line_mask(new uint8_t[orthstride + coordinate_index(c.topo.dim_y)*linestride[1]]{}) {
        compute_line_masks();
    }

    size_t score(size_y yc, size_x xc, size_y y0, size_y y1, size_x x0, size_x x1) const {
        return min(
            get_line_score(0, vert(xc), vert(y0), vert(y1)),
            get_line_score(1, horz(yc), horz(x0), horz(x1))
        );
    }

    void inflate(size_y yc, size_x xc, size_y y0, size_y y1, size_x x0, size_x x1,
                 vector<vector<size_t>> &emb) const {
        uint8_t k0 = get_line_mask(0, vert(xc), vert(y0), vert(y1));
        uint8_t k1 = get_line_mask(1, horz(yc), horz(x0), horz(x1));
        while (k0 && k1) {
            emb.emplace_back(0);
            vector<size_t> &chain = emb.back();
            cells.topo.construct_line(0, vert(xc), vert(y0), vert(y1), first_bit[k0], chain);
            cells.topo.construct_line(1, horz(yc), horz(x0), horz(x1), first_bit[k1], chain);
            k0 ^= mask_bit[first_bit[k0]];
            k1 ^= mask_bit[first_bit[k1]];
        }
    }

    void inflate(size_y y0, size_y y1, size_x x0, size_x x1,
                 vector<vector<size_t>> &emb) const {
        inflate(0, y0, y1, x0, x1, emb);
        inflate(1, y0, y1, x0, x1, emb);
    }

    void inflate(bool u, size_y y0, size_y y1, size_x x0, size_x x1,
                 vector<vector<size_t>> &emb) const {
        size_w w0, w1;
        size_z z0, z1;
        if(u) {
            w0 = horz(y0); w1 = horz(y1);
            z0 = horz(x0); z1 = horz(x1);
        } else {
            w0 = vert(x0); w1 = vert(x1);
            z0 = vert(y0); z1 = vert(y1);
        }
        for(size_w w = w0; w <= w1; w++) {
            uint8_t k = get_line_mask(u, w, z0, z1);
            while(k) {
                emb.emplace_back(0);
                vector<size_t> &chain = emb.back();
                cells.topo.construct_line(u, w, z0, z1, first_bit[k], chain);
                k ^= mask_bit[first_bit[k]];
            }
        }
    }

    size_t length(size_y yc, size_x xc, size_y y0, size_y y1, size_x x0, size_x x1) const {
        return cells.topo.line_length(0, vert(xc), vert(y0), vert(y1)) + cells.topo.line_length(1, horz(yc), horz(x0), horz(x1));
    }

    inline uint8_t get_line_score(bool u, size_w w, size_z z0, size_z z1) const {
        return popcount[get_line_mask(u, w, z0, z1)];
    }

  private:
  
    inline uint8_t &get_line_mask(bool u, size_w w, size_z z0, size_z z1) const {
        minorminer_assert(u?(horz(w)<cells.topo.dim_y):(vert(w)<cells.topo.dim_x));
        minorminer_assert(z0 <= z1);
        minorminer_assert(u?(horz(z1)<cells.topo.dim_x):(vert(z1)<cells.topo.dim_y));
        size_t index = coordinate_converter::bundle_cache_index(u, w, z0, z1, orthstride, linestride[u]);
        minorminer_assert(index < orthstride + coordinate_index(cells.topo.dim_y)*linestride[1]);
        return line_mask[index];
    }

    void compute_line_masks() {
        uint8_t *mask = line_mask;
        struct {bool u; size_w dim_w; size_z dim_z; } sides[2] = {
            {0, vert(cells.topo.dim_x), vert(cells.topo.dim_y)},
            {1, horz(cells.topo.dim_y), horz(cells.topo.dim_x)},
        };
        for (auto side : sides) {
            for (size_w w = 0; w < side.dim_w; w++) {
                for (size_z z = 0; z < side.dim_z; z++) {
                    mask+= coordinate_index(z);
                    minorminer_assert(mask+coordinate_index(z) == &get_line_mask(side.u, w, z, z));
                    uint8_t m = mask[coordinate_index(z)] = cells.qmask(side.u, w, z);
                    for(size_z z0 = z; z0-->0_z;) {
                        minorminer_assert(mask+coordinate_index(z0) == &get_line_mask(side.u, w, z0, z));
                        m = mask[coordinate_index(z0)] = m & cells.emask(side.u, w, z0+1_z);
                    }
                }
                mask += coordinate_index(side.dim_z);
            }
        }
        minorminer_assert(mask == line_mask + orthstride + coordinate_index(cells.topo.dim_y)*linestride[1]);
    }
};

}
