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

template<char C>
size_t binom(coordinate_base<C> c) {
    return binom(c.index());
}

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
                 orthstride((c.topo.dim_x*linestride[0]).index()),
                 line_mask(new uint8_t[orthstride + c.topo.dim_x.index()*linestride[1]]{}) {
        compute_line_masks();
    }

    size_t score(size_y yc, size_x xc, size_y y0, size_y y1, size_x x0, size_x x1) const {
        return min(
            get_line_score(0, xc.vert(), y0.vert(), y1.vert()),
            get_line_score(1, yc.horz(), x0.horz(), x1.horz())
        );
    }

    void inflate(size_y yc, size_x xc, size_y y0, size_y y1, size_x x0, size_x x1,
                 vector<vector<size_t>> &emb) const {
        uint8_t k0 = get_line_mask(0, xc.vert(), y0.vert(), y1.vert());
        uint8_t k1 = get_line_mask(1, yc.horz(), x0.horz(), x1.horz());
        while (k0 && k1) {
            emb.emplace_back(0);
            vector<size_t> &chain = emb.back();
            cells.topo.construct_line(0, xc.vert(), y0.vert(), y1.vert(), first_bit[k0], chain);
            cells.topo.construct_line(1, yc.horz(), x0.horz(), x1.horz(), first_bit[k1], chain);
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
            w0 = y0.horz(); w1 = y1.horz();
            z0 = x0.horz(); z1 = x1.horz();
        } else {
            w0 = x0.vert(); w1 = x1.vert();
            z0 = y0.vert(); z1 = y1.vert();
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
        return cells.topo.line_length(xc, y0, y1) + cells.topo.line_length(yc, x0, x1);
    }

    inline uint8_t get_line_score(bool u, size_w w, size_z z0, size_z z1) const {
        return popcount[get_line_mask(u, w, z0, z1)];
    }

    inline uint8_t get_line_score(size_x x, size_y y0, size_y y1) const {
        return get_line_score(0, x.vert(), y0.vert(), y1.vert());
    }

    inline uint8_t get_line_score(size_y y, size_x x0, size_x x1) const {
        return get_line_score(1, y.horz(), x0.horz(), x1.horz());
    }

  private:
    inline uint8_t &get_line_mask(bool u, size_w w, size_z z0, size_z z1) const {
        minorminer_assert(u?(w.horz()<cells.topo.dim_y):(w.vert()<cells.topo.dim_x));
        minorminer_assert(z0 <= z1);
        minorminer_assert(u?(z1.horz()<cells.topo.dim_x):(z1.vert()<cells.topo.dim_y));
        size_t index = coordinate_converter::sum(u*orthstride, w*linestride[u], binom(z1.index()), z0);
        return line_mask[index];
    }

    void compute_line_masks() {

        for (size_w w = 0; w < cells.topo.dim_x.vert(); w++) {
            uint8_t *t_base = line_mask + coordinate_converter::sum(w*linestride[0]);
            for (size_z z = 0; z < cells.topo.dim_y.vert(); z++) {
                uint8_t *t = t_base + binom(z);
                uint8_t m = t[z.index()] = cells.qmask(0, w, z);
                for(size_z z0 = z; z0-->size_z(0);) {
                    m = t[z0.index()] = m & cells.emask(0, w, z0+1u);
                    minorminer_assert(t+z0.index() == &get_line_mask(0, w, z0, z));
                }
            }
        }
        for (size_w w = 0; w < cells.topo.dim_y.horz(); w++) {
            uint8_t *t_base = line_mask + coordinate_converter::sum(orthstride, w*linestride[1]);
            for (size_z z = 0; z < cells.topo.dim_x.horz(); z++) {
                uint8_t *t = t_base + binom(z);
                uint8_t m = t[z.index()] = cells.qmask(1, w, z);
                for(size_z z0 = z; z0-->size_z(0);) {
                    m = t[z0.index()] = m & cells.emask(1, w, z0+1u);
                    minorminer_assert(t+z0.index() == &get_line_mask(1, w, z0, z));
                }
            }
        }


    }
};

}
