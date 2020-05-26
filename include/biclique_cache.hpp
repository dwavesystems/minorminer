#pragma once
#include "util.hpp"

namespace busclique {

class yieldcache {
  public:
    const size_t rows;
    const size_t cols;
  private:
    size_t *mem;
  public:
    yieldcache(size_t r, size_t c, size_t *m) : rows(r), cols(c), mem(m) {};
    size_t get(size_t y, size_t x, size_t u) {
        Assert(y < rows);
        Assert(x < cols);
        Assert(u < 2);
        return mem[(y*cols + x)*2 + u];
    }
    void set(size_t y, size_t x, size_t u, size_t score) {
        Assert(y < rows);
        Assert(x < cols);
        Assert(u < 2);
        mem[(y*cols + x)*2 + u] = score;
    }
    
};

template<typename topo_spec>
class biclique_cache {
  public:
    biclique_cache(const biclique_cache&) = delete; 
    biclique_cache(biclique_cache &&) = delete;
  private:
    const cell_cache<topo_spec> &cells;
    size_t *mem;

    
    size_t memrows(size_t h) const {
        return cells.topo.dim[0] - h + 1;
    }
    size_t memcols(size_t w) const {
        return cells.topo.dim[1] - w + 1;
    }

    size_t memsize(size_t h, size_t w) const {
        return 2 * memrows(h) * memcols(w);
    }

    size_t memsize() const {
        size_t size = 0;
        for(size_t h = 0; h <= cells.topo.dim[0]; h++)
            for(size_t w = 0; w <= cells.topo.dim[1]; w++)
                size += memsize(h, w) + 1;
        return size;
    }

    size_t mem_addr(size_t h, size_t w) const {
        return (h-1)*cells.topo.dim[1] + (w-1);
    }
        

    void make_access_table() {
        size_t offset = cells.topo.dim[0]*cells.topo.dim[1];
        for(size_t h = 1; h <= cells.topo.dim[0]; h++)
            for(size_t w = 1; w <= cells.topo.dim[1]; w++) {        
                mem[mem_addr(h,w)] = offset;
                offset += memsize(h, w);
            }
    }

  public:
    yieldcache get(size_t h, size_t w) const {
        Assert(1 <= h);
        Assert(h <= cells.topo.dim[0]);
        Assert(1 <= w);
        Assert(w <= cells.topo.dim[1]);
        return yieldcache(memrows(h), memcols(w), mem + mem[mem_addr(h,w)]);
    }

  private:
    void compute_cache(const bundle_cache<topo_spec> &bundles) {
        for(size_t h = 1; h <= cells.topo.dim[0]; h++) {
            {
                size_t w = 1;
                yieldcache next = get(h, w);
                for(size_t y0 = 0; y0 <= cells.topo.dim[0]-h; y0++)
                    for(size_t x0 = 0; x0 <= cells.topo.dim[1]-w; x0++)
                        next.set(y0, x0, 0, bundles.get_line_score(0, x0, y0, y0+h-1));
            }
            for(size_t w = 2; w < cells.topo.dim[1]; w++) {
                yieldcache prev = get(h, w-1);
                yieldcache next = get(h, w);
                for(size_t y0 = 0; y0 <= cells.topo.dim[0]-h; y0++) {
                    size_t score = prev.get(y0, 0, 0);
                    score += bundles.get_line_score(0, w-1, y0, y0+h-1);
                    next.set(y0, 0, 0, score);
                    for(size_t x0 = 1; x0 <= cells.topo.dim[1]-w; x0++) {
                        score -= bundles.get_line_score(0, x0-1, y0, y0+h-1);
                        score += bundles.get_line_score(0, x0+w-1, y0, y0+h-1);
                        next.set(y0, x0, 0, score);
                    }
                }
            }
        }
        for(size_t w = 1; w <= cells.topo.dim[0]; w++) {
            {
                size_t h = 1;
                yieldcache next = get(h, w);
                for(size_t y0 = 0; y0 <= cells.topo.dim[0]-h; y0++)
                    for(size_t x0 = 0; x0 <= cells.topo.dim[1]-w; x0++)
                        next.set(y0, x0, 1, bundles.get_line_score(1, y0, x0, x0+w-1));
            }
            for(size_t h = 2; h < cells.topo.dim[0]; h++) {
                yieldcache prev = get(h-1, w);
                yieldcache next = get(h, w);
                for(size_t x0 = 0; x0 <= cells.topo.dim[1]-w; x0++) {
                    size_t score = prev.get(0, x0, 1);
                    score += bundles.get_line_score(1, h-1, x0, x0+w-1);
                    next.set(0, x0, 1, score);
                    for(size_t y0 = 1; y0 <= cells.topo.dim[0]-h; y0++) {
                        score -= bundles.get_line_score(1, y0-1, x0, x0+w-1);
                        score += bundles.get_line_score(1, y0+h-1, x0, x0+w-1);
                        next.set(y0, x0, 1, score);
                    }
                }
            }
        }
    }

  public:
    biclique_cache(const cell_cache<topo_spec> &c, const bundle_cache<topo_spec> &b) :
        cells(c), mem(new size_t[memsize()]{}) {
        make_access_table();
        compute_cache(b);
    }
    ~biclique_cache() {
        if(mem != nullptr) { delete[] mem; mem = nullptr; }
    }

    std::pair<size_t, size_t> score(size_t y0, size_t y1, size_t x0, size_t x1) const {
        Assert(y0 <= y1);
        Assert(x0 <= x1);
        Assert(y1 < cells.topo.dim[0]);
        Assert(x1 < cells.topo.dim[1]);
        size_t h = y1 - y0 + 1;
        size_t w = x1 - x0 + 1;
        yieldcache cache = get(h, w);
        return std::make_pair<size_t, size_t>(cache.get(y0, x0, 0), cache.get(y0, x0, 1));
    }
};

}

