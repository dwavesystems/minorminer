#pragma once
#include "util.hpp"
#include "cell_cache.hpp"

namespace busclique {

size_t n_segments(size_t x) { return (x*x+x)/2; }

template<typename topo_spec>
class bundle_cache {
    const cell_cache<topo_spec> &cells;
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
                 linestride{n_segments(c.topo.dim[0]), n_segments(c.topo.dim[1])},
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
            cells.construct_line(0, xc, y0, y1, first_bit[k0], chain);
            cells.construct_line(1, yc, x0, x1, first_bit[k1], chain);
            k0 ^= mask_bit[first_bit[k0]];
            k1 ^= mask_bit[first_bit[k1]];
        }
    }

    size_t length(size_t yc, size_t xc, size_t y0, size_t y1, size_t x0, size_t x1) const {
        return cells.line_length(0, xc, y0, y1) + cells.line_length(1, yc, x0, x1);
    }

  private:
    inline uint8_t get_line_score(size_t u, size_t w, size_t z0, size_t z1) const {
        return popcount[get_line_mask(u, w, z0, z1)];
    }

    inline uint8_t get_line_mask(size_t u, size_t w, size_t z0, size_t z1) const {
        Assert(u < 2);
        Assert(w < cells.topo.dim[1-u]);
        Assert(z0 <= z1);
        Assert(z1 < cells.topo.dim[u]);
        return line_mask[u*orthstride + w*linestride[u] + (z1*z1+z1)/2 + z0];
    }

    void compute_line_masks() {
        for(size_t u = 0; u < 2; u++) {
            for (size_t w = 0; w < cells.topo.dim[1-u]; w++) {
                for (size_t z = 0; z < cells.topo.dim[u]; z++) {
                    uint8_t *t = line_mask + u*orthstride + w*linestride[u] + (z*z+z)/2;
                    uint8_t m = t[z] = cells.qmask(u, w, z);
                    for(size_t z0 = z; z0--;)
                        m = t[z0] = m & cells.emask(u, w, z0+1);
                }
            }
        }
    }
};

}
