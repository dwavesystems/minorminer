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

namespace busclique {

class yieldcache {
  public:
    const size_t rows;
    const size_t cols;
  private:
    size_t *mem;
  public:
    yieldcache(size_t r, size_t c, size_t *m) : rows(r), cols(c), mem(m) {};
    size_t get(size_t y, size_t x, size_t u) const {
        minorminer_assert(y < rows);
        minorminer_assert(x < cols);
        minorminer_assert(u < 2);
        return mem[(y*cols + x)*2 + u];
    }
    void set(size_t y, size_t x, size_t u, size_t score) {
        minorminer_assert(y < rows);
        minorminer_assert(x < cols);
        minorminer_assert(u < 2);
        mem[(y*cols + x)*2 + u] = score;
    }
    
};

template<typename topo_spec>
class biclique_cache {
  public:
    biclique_cache(const biclique_cache&) = delete;
    biclique_cache(biclique_cache &&) = delete;
    const cell_cache<topo_spec> &cells;
  private:
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
        minorminer_assert(1 <= h);
        minorminer_assert(h <= cells.topo.dim[0]);
        minorminer_assert(1 <= w);
        minorminer_assert(w <= cells.topo.dim[1]);
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
            for(size_t w = 2; w <= cells.topo.dim[1]; w++) {
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
        for(size_t w = 1; w <= cells.topo.dim[1]; w++) {
            {
                size_t h = 1;
                yieldcache next = get(h, w);
                for(size_t y0 = 0; y0 <= cells.topo.dim[0]-h; y0++)
                    for(size_t x0 = 0; x0 <= cells.topo.dim[1]-w; x0++)
                        next.set(y0, x0, 1, bundles.get_line_score(1, y0, x0, x0+w-1));
            }
            for(size_t h = 2; h <= cells.topo.dim[0]; h++) {
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
        minorminer_assert(y0 <= y1);
        minorminer_assert(x0 <= x1);
        minorminer_assert(y1 < cells.topo.dim[0]);
        minorminer_assert(x1 < cells.topo.dim[1]);
        size_t h = y1 - y0 + 1;
        size_t w = x1 - x0 + 1;
        yieldcache cache = get(h, w);
        return std::make_pair<size_t, size_t>(cache.get(y0, x0, 0), cache.get(y0, x0, 1));
    }
};

template<typename topo_spec>
class biclique_yield_cache {
    using bound_t = std::tuple<size_t, size_t, size_t, size_t>;

  public:
    const cell_cache<topo_spec> &cells;
    const bundle_cache<topo_spec> &bundles;

  private:
    const size_t rows, cols;
    vector<vector<size_t>> chainlength;
    vector<vector<bound_t>> biclique_bounds;

  public:
    biclique_yield_cache(const biclique_yield_cache&) = delete; 
    biclique_yield_cache(biclique_yield_cache &&) = delete;

    biclique_yield_cache(const cell_cache<topo_spec> &c,
                         const bundle_cache<topo_spec> &b, 
                         const biclique_cache<topo_spec> &bicliques) :
        cells(c),
        bundles(b),

        //note: the role of rows and columns is reversed, because the indices
        //are chainlengths in a given direction (not the number of chains)
        rows(cells.topo.dim[1]*cells.topo.shore),
        cols(cells.topo.dim[0]*cells.topo.shore),

        chainlength(rows, vector<size_t>(cols, 0)),
        biclique_bounds(rows, vector<bound_t>(cols, bound_t(0,0,0,0))) {
        compute_cache(bicliques);
    }
  private:
    void compute_cache(const biclique_cache<topo_spec> &bicliques) {
        for(size_t h = 1; h <= cells.topo.dim[0]; h++) {
            for(size_t w = 1; w <= cells.topo.dim[1]; w++) {
                auto cache = bicliques.get(h, w);
                for(size_t y = 0; y < cache.rows; y++) {
                    for(size_t x = 0; x < cache.cols; x++) {
                        size_t s0 = cache.get(y, x, 0);
                        size_t s1 = cache.get(y, x, 1);
                        if (s0 == 0 || s1 == 0) continue;
                        minorminer_assert(s0-1 < rows);
                        minorminer_assert(s1-1 < cols);
                        size_t maxlen = cells.topo.biclique_length(y, y+h-1, x, x+w-1);
                        size_t prevlen = chainlength[s0-1][s1-1];
                        if(prevlen == 0 || prevlen > maxlen) {
                            chainlength[s0-1][s1-1] = maxlen;
                            biclique_bounds[s0-1][s1-1] = bound_t(y, y+h-1, x, x+w-1);
                        }
                    }
                }
            }
        }
    }

  public:
    class iterator {
        size_t s0, s1;
        const size_t &rows, &cols;
        const vector<vector<size_t>> &chainlength;
        const vector<vector<bound_t>> &bounds;
        const bundle_cache<topo_spec> &bundles;
        void adv() {
            while(s0 < rows && chainlength[s0][s1] == 0) inc();
        }
        bool inc() {
            if(s0 >= rows) return false;
            s1++;
            if(s1 == cols) { s1 = 0; s0++; }
            return true;
        }
      public:
        iterator(size_t _s0, size_t _s1, const size_t &r, const size_t &c,
                 const vector<vector<size_t>> &cl,
                 const vector<vector<bound_t>> &_bounds,
                 const bundle_cache<topo_spec> &_bundles) :
                 s0(_s0), s1(_s1), rows(r), cols(c), chainlength(cl),
                 bounds(_bounds), bundles(_bundles) { adv(); }

        iterator operator++() { iterator i = *this; if (inc()) adv(); return i; }
        iterator operator++(int) { if(inc()) adv(); return *this; }
        std::tuple<size_t, size_t, size_t, vector<vector<size_t>>> operator*() { 
            bound_t z = bounds[s0][s1];
            size_t cl = chainlength[s0][s1];
            minorminer_assert(cl > 0); 
            vector<vector<size_t>> emb;
            bundles.inflate(std::get<0>(z), std::get<1>(z), 
                            std::get<2>(z), std::get<3>(z), emb);
            return std::make_tuple(s0+1, s1+1, cl, emb);
        }
        bool operator==(const iterator& rhs) { return s0 == rhs.s0 && s1 == rhs.s1; }
        bool operator!=(const iterator& rhs) { return !operator==(rhs); }
    };

    iterator begin() const {
        return iterator(0, 0, rows, cols, chainlength, biclique_bounds, bundles);
    }
    iterator end() const {
        return iterator(rows, 0, rows, cols, chainlength, biclique_bounds, bundles);
    }
};

}

