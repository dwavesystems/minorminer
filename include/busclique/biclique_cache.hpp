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
    const size_y rows;
    const size_x cols;
  private:
    size_t *mem;
    size_t index(size_y y, size_x x, bool u) const {
        return coordinate_converter::cell_index(y, x, u, rows, cols);
    }
  public:
    yieldcache(size_y r, size_x c, size_t *m) : rows(r), cols(c), mem(m) {};
    size_t get(size_y y, size_x x, bool u) const {
        return mem[index(y, x, u)];
    }
    void set(size_y y, size_x x, bool u, size_t score) {
        mem[index(y, x, u)] = score;
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

    size_y memrows(size_y h) const {
        return cells.topo.dim_y - h + 1_y;
    }
    size_x memcols(size_x w) const {
        return cells.topo.dim_x - w + 1_x;
    }

    size_t memsize(size_y h, size_x w) const {
        return coordinate_converter::product(2, memrows(h), memcols(w));
    }

    size_t memsize() const {
        size_t size = 0;
        for(size_y h = 0; h <= cells.topo.dim_y; h++)
            for(size_x w = 0; w <= cells.topo.dim_x; w++)
                size += memsize(h, w) + 1;
        return size;
    }

    size_t mem_addr(size_y h, size_x w) const {
        return coordinate_converter::grid_index(h-1_y, w-1_x, cells.topo.dim_y, cells.topo.dim_x);
    }
        

    void make_access_table() {
        size_t offset = coordinate_converter::product(cells.topo.dim_y, cells.topo.dim_x);
        for(size_y h = 1; h <= cells.topo.dim_y; h++)
            for(size_x w = 1; w <= cells.topo.dim_x; w++) {        
                mem[mem_addr(h,w)] = offset;
                offset += memsize(h, w);
            }
    }

  public:
    yieldcache get(size_y h, size_x w) const {
        minorminer_assert(0_y < h);
        minorminer_assert(h <= cells.topo.dim_y);
        minorminer_assert(0_x < w);
        minorminer_assert(w <= cells.topo.dim_x);
        return yieldcache(memrows(h), memcols(w), mem + mem[mem_addr(h,w)]);
    }

  private:
    void compute_cache(const bundle_cache<topo_spec> &bundles) {
        for(size_y h = 1; h <= cells.topo.dim_y; h++) {
            {
                size_x w = 1;
                yieldcache next = get(h, w);
                for(size_y y0 = 0; y0 <= cells.topo.dim_y-h; y0++)
                    for(size_x x0 = 0; x0 <= cells.topo.dim_x-w; x0++)
                        next.set(y0, x0, 0, bundles.get_line_score(0, vert(x0), vert(y0), vert(y0+h-1_y)));
            }
            for(size_x w = 2; w <= cells.topo.dim_x; w++) {
                yieldcache prev = get(h, w-1_x);
                yieldcache next = get(h, w);
                for(size_y y0 = 0; y0 <= cells.topo.dim_y-h; y0++) {
                    size_t score = prev.get(y0, 0, 0);
                    score += bundles.get_line_score(0, vert(w-1_x), vert(y0), vert(y0+h-1_y));
                    next.set(y0, 0, 0, score);
                    for(size_x x0 = 1; x0 <= cells.topo.dim_x-w; x0++) {
                        score -= bundles.get_line_score(0, vert(x0-1_x), vert(y0), vert(y0+h-1_y));
                        score += bundles.get_line_score(0, vert(x0+w-1_x), vert(y0), vert(y0+h-1_y));
                        next.set(y0, x0, 0, score);
                    }
                }
            }
        }
        for(size_x w = 1; w <= cells.topo.dim_x; w++) {
            {
                size_y h = 1;
                yieldcache next = get(h, w);
                for(size_y y0 = 0; y0 <= cells.topo.dim_y-h; y0++)
                    for(size_x x0 = 0; x0 <= cells.topo.dim_x-w; x0++)
                        next.set(y0, x0, 1, bundles.get_line_score(1, horz(y0), horz(x0), horz(x0+w-1_x)));
            }
            for(size_y h = 2; h <= cells.topo.dim_y; h++) {
                yieldcache prev = get(h-1_y, w);
                yieldcache next = get(h, w);
                for(size_x x0 = 0; x0 <= cells.topo.dim_x-w; x0++) {
                    size_t score = prev.get(0, x0, 1);
                    score += bundles.get_line_score(1, horz(h-1_y), horz(x0), horz(x0+w-1_x));
                    next.set(0, x0, 1, score);
                    for(size_y y0 = 1; y0 <= cells.topo.dim_y-h; y0++) {
                        score -= bundles.get_line_score(1, horz(y0-1_y), horz(x0), horz(x0+w-1_x));
                        score += bundles.get_line_score(1, horz(y0+h-1_y), horz(x0), horz(x0+w-1_x));
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

    std::pair<size_t, size_t> score(size_y y0, size_y y1, size_x x0, size_x x1) const {
        minorminer_assert(y0 <= y1);
        minorminer_assert(x0 <= x1);
        minorminer_assert(y1 < cells.topo.dim_y);
        minorminer_assert(x1 < cells.topo.dim_x);
        size_y h = y1 - y0 + 1_y;
        size_x w = x1 - x0 + 1_x;
        yieldcache cache = get(h, w);
        return std::make_pair<size_t, size_t>(cache.get(y0, x0, 0), cache.get(y0, x0, 1));
    }
};

template<typename topo_spec>
class biclique_yield_cache {
    using bound_t = std::tuple<size_y, size_y, size_x, size_x>;

  public:
    const cell_cache<topo_spec> &cells;
    const bundle_cache<topo_spec> &bundles;

  private:
    //note: the role of rows and columns is reversed, because the indices
    //are chainlengths in a given direction (not the number of chains)
    const size_x rows;
    const size_y cols;
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

        rows(cells.topo.dim_x*cells.topo.shore),
        cols(cells.topo.dim_y*cells.topo.shore),

        chainlength(coordinate_index(rows), vector<size_t>(coordinate_index(cols), 0)),
        biclique_bounds(coordinate_index(rows), vector<bound_t>(coordinate_index(cols), bound_t(0,0,0,0))) {
        compute_cache(bicliques);
    }
  private:
    void compute_cache(const biclique_cache<topo_spec> &bicliques) {
        for(size_y h = 1; h <= cells.topo.dim_y; h++) {
            for(size_x w = 1; w <= cells.topo.dim_x; w++) {
                auto cache = bicliques.get(h, w);
                for(size_y y = 0; y < cache.rows; y++) {
                    for(size_x x = 0; x < cache.cols; x++) {
                        size_t s0 = cache.get(y, x, 0);
                        size_t s1 = cache.get(y, x, 1);
                        if (s0 == 0 || s1 == 0) continue;
                        minorminer_assert(size_x(s0-1) < rows);
                        minorminer_assert(size_y(s1-1) < cols);
                        size_t maxlen = cells.topo.biclique_length(y, y+h-1_y, x, x+w-1_x);
                        size_t prevlen = chainlength[s0-1][s1-1];
                        if(prevlen == 0 || prevlen > maxlen) {
                            chainlength[s0-1][s1-1] = maxlen;
                            biclique_bounds[s0-1][s1-1] = bound_t(y, y+h-1_y, x, x+w-1_x);
                        }
                    }
                }
            }
        }
    }

  public:
    class iterator {
        size_t s0, s1;
        const size_x &rows;
        const size_y &cols;
        const vector<vector<size_t>> &chainlength;
        const vector<vector<bound_t>> &bounds;
        const bundle_cache<topo_spec> &bundles;
        void adv() {
            while(size_x(s0) < rows && chainlength[s0][s1] == 0) inc();
        }
        bool inc() {
            if(size_x(s0) >= rows) return false;
            s1++;
            if(size_y(s1) == cols) { s1 = 0; s0++; }
            return true;
        }
      public:
        iterator(size_t _s0, size_t _s1, const size_x &r, const size_y &c,
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
        return iterator(coordinate_index(rows), 0, rows, cols, chainlength, biclique_bounds, bundles);
    }
};

}

