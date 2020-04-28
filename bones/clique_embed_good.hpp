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

namespace chimera_clique {

using std::numeric_limits;
using std::vector;
using std::pair;
using std::min;
using std::max;

void Assert(bool thing) {
//    if(!thing) throw std::exception();
}

enum corner : size_t { 
    NW = 1,
    NE = 2,
    SW = 4,
    SE = 8,
    shift = 4,
    mask = 15
};
using corner::NW;
using corner::NE;
using corner::SW; 
using corner::SE; 

const size_t null_node = numeric_limits<size_t>::max();

class chimera_lines {
  public:
    const size_t dim;
    const size_t shore;
  private:
    vector<bool> nodemask;
    vector<bool> edgemask;

  public:
    chimera_lines(size_t d, size_t s, vector<size_t> nodes,
                               vector<pair<size_t, size_t>> edges) : dim(d),
                               shore(s), nodemask(d*d*s*2, false),
                               edgemask(d*d*s*2, false) {
        if(shore > 256)
            throw std::exception();
        for(auto &q: nodes) nodemask[q] = true;
        size_t stride_0 = dim*shore*2;
        size_t stride_1 = shore*2;
        for(auto &pq: edges) {
            size_t root = ext_root(pq.first, pq.second, stride_0, stride_1);
            if (root != null_node)
                edgemask[root] = true;
        }
    }

    chimera_lines(size_t d, size_t s,
                               vector<pair<size_t, size_t>> edges) : dim(d),
                               shore(s), nodemask(d*d*s*2, false),
                               edgemask(d*d*s*2, false) {
        size_t stride_0 = dim*shore*2;
        size_t stride_1 = shore*2;
        for(auto &pq: edges) {
            nodemask[pq.first] = true;
            nodemask[pq.second] = true;
            size_t root = ext_root(pq.first, pq.second, 
                                        stride_0, stride_1);
            if (root != null_node)
                edgemask[root] = true;
        }
    }


    bool has_qubit (size_t y, size_t x, size_t u, size_t k) const {
        Assert(y < dim);
        Assert(x < dim);
        Assert(u < 2);
        Assert(k < shore);
        return nodemask[k + shore*(u + 2*(x + dim*y))];
    }

    //! returns true if the edge (y, x, u, k) ~ (y-1+u, x-u, u, k)
    //! is contained within this graph
    bool has_ext (size_t y, size_t x, size_t u, size_t k) const {
        Assert(y < dim);
        Assert(x < dim);
        Assert(u < 2);
        Assert(k < shore);
        return edgemask[k + shore*(u + 2*(x + dim*y))];
    }

  private:
    inline size_t ext_root(size_t p, size_t q, size_t stride_0, 
                                size_t stride_1) {
        size_t puk = p%(shore*2);
        size_t quk = q%(shore*2);

        if (puk != quk)
            return null_node;

        size_t stride = (puk < shore)?stride_0:stride_1;

        if (p + stride == q)
            return q;
        if (q + stride == p)
            return p;

        return null_node;
    }

  public:
    void inflate_bundle(size_t y0, size_t y1, size_t x0, size_t x1,
                        vector<vector<size_t>> &chains) const {
        vector<bool> h_mask = scan_line(1, y0, x0, x1);
        vector<bool> v_mask = scan_line(0, x0, y0, y1);
        vector<size_t> h_k;
        vector<size_t> v_k;
        for(size_t k = 0; k < shore; k++) {
            if(h_mask[k]) h_k.push_back(k);
            if(v_mask[k]) v_k.push_back(k);
        }
        for(size_t i = 0; i < min(h_k.size(), v_k.size()); i++) {
            chains.emplace_back(0);
            inflate_line(1, y0, x0, x1, h_k[i], chains.back());
            inflate_line(0, x0, y0, y1, v_k[i], chains.back());
        }
    }

  protected:
    void inflate_line(size_t u, size_t w, size_t z0, size_t z1, size_t k,
                      vector<size_t> &chain) const {
        size_t p[2], &z = p[u], &y = p[0], &x = p[1];
        p[1-u] = w;
        if(z0 > z1) std::swap(z0, z1);
        for(z = z0; z <= z1; z++)
            chain.push_back(k + shore*(u + 2*(x + dim*y)));
    }

    vector<bool> scan_line(size_t u, size_t w, size_t z0, size_t z1) const {
        size_t p[2], &z = p[u], &y = p[0], &x = p[1];
        p[1-u] = w;
        vector<bool> K(shore, true);
        if(z0 > z1) std::swap(z0, z1);
        z = z0;
        for(size_t k = 0; k < shore; k++)
            K[k] = has_qubit(y, x, u, k);
        while(++z <= z1)
            for(size_t k = 0; k < shore; k++)
                K[k] = K[k] & has_qubit(y, x, u, k) & has_ext(y, x, u, k);
        return K;
    }
};

class bundle_cache {
    const size_t dim;
    const size_t linestride;
    const size_t orthstride;
    uint8_t *line_score;

  public:
    ~bundle_cache() {
        if (line_score != nullptr) {
            delete [] line_score;
            line_score = nullptr;
        }
    }
    bundle_cache(const chimera_lines &chim) :
                 dim(chim.dim),
                 linestride((dim*dim+dim)/2),
                 orthstride(dim*linestride),
                 line_score(new uint8_t[2*orthstride]) {
        compute_line_scores(chim);
    }

    size_t ell_score(size_t y0, size_t y1, size_t x0, size_t x1) const {
        return min(get_hline_score(y0, min(x0, x1), max(x0, x1)),
                   get_vline_score(x0, min(y0, y1), max(y0, y1)));
    }

    inline size_t get_hline_score(size_t y, size_t x0, size_t x1) const {
        Assert(y < dim);
        Assert(x0 <= x1);
        Assert(x1 < dim);
        return line_score[orthstride + y*linestride + (x1*x1+x1)/2 + x0];
    }

    inline size_t get_vline_score(size_t x, size_t y0, size_t y1) const {
        Assert(x < dim);
        Assert(y0 <= y1);
        Assert(y1 < dim);
        return line_score[x*linestride + (y1*y1+y1)/2 + y0];
    }

  private:
    void compute_line_scores(const chimera_lines &chim) {
        size_t shoreline[chim.shore];
        size_t scanline[chim.dim];
        for(size_t u = 0; u < 2; u++) {
            size_t p[2];
            size_t &y = p[0]; size_t &z = p[u];
            size_t &x = p[1]; size_t &w = p[1-u];
            for (w = 0; w < chim.dim; w++) {
                for (z = 0; z < chim.dim; z++) {
                    std::fill(scanline, scanline+z+1, 0);
                    for(size_t k = 0; k < chim.shore; k++) {
                        if (chim.has_ext(y, x, u, k)) {
                            // we're on a good qubit connected to a line; bump
                            // up the whole thing starting at shoreline[k]
                            for(size_t z0 = shoreline[k]; z0 <= z; z0++)
                                scanline[z0]++;
                        } else {
                            if(chim.has_qubit(y, x, u, k)) {
                                // start up a new line -- this is where we 
                                // actually initialize shoreline (sorry)
                                shoreline[k] = z;
                                scanline[z]++;
                            } else {
                                // ded
                                shoreline[k] = null_node;
                            }
                        }
                    }
                    //write the current line into cache
                    size_t offset = u*orthstride + w*linestride + (z*z+z)/2;
                    std::copy(scanline, scanline+z+1, line_score+offset);
                }
            }
        }
    }    
};


class rectangle_cache {
    friend class solution_iterator;
  public:
    class maxcache {
      public:
        const size_t rows;
        const size_t cols;
      private:
        size_t *mem;
      public:
        maxcache(size_t r, size_t c, size_t *m) : rows(r), cols(c), mem(m) {};
        void setmax(size_t y, size_t x, size_t score, corner c) { 
            Assert(y < rows);
            Assert(x < cols);
            score = (score << corner::shift) | c;
            if(score > mem[y*cols+x])
                mem[y*cols+x] = score;
        }
        size_t score(size_t y, size_t x) const {
            Assert(y < rows);
            Assert(x < cols);
            return mem[y*cols + x] >> corner::shift;
        }
        size_t corner(size_t y, size_t x) const {
            Assert(y < rows);
            Assert(x < cols);
            return mem[y*cols + x] & corner::mask;
        }
    };
    
  private:
    bundle_cache &bundles;
    size_t dim;
    size_t width;
    size_t memsize(size_t i) const {
        return (dim-i)*(dim-width+i+2);
    }

    size_t *mem;
    size_t memsize() const {
        size_t size = 0;
        for(size_t i = 0; i < width-1; i++)
            size += memsize(i);
        return size + width;
    }

  public:
    rectangle_cache(size_t d, size_t w, bundle_cache &b) : 
            bundles(b),
            dim(d),
            width(w),
            mem(new size_t[memsize()]{}) {
        mem[0] = width;
        for(size_t i = 1; i < width-1; i++)
            mem[i] = mem[i-1] + memsize(i-1);
        compute_cache();
    }

    ~rectangle_cache() {
        if (mem != nullptr) {
            delete [] mem;
            mem = nullptr;
        }
    }

    maxcache get(size_t i) {
        Assert(i < width-1);
        return maxcache(dim-i, dim-width+i+2, mem + mem[i]);
    }

    void print() {
        for(size_t i = 0; i < width-1; i++) {
            maxcache m = get(i);
            std::cout << mem[i] << ':' << memsize(i) << "?"<< std::endl;
            for(size_t y = 0; y < m.rows; y++) {
                for(size_t x = 0; x < m.cols; x++) {
                    std::cout << m.score(y, x) << '~' << (m.corner(y, x)) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
  private:       
    void compute_cache() {
        {
            maxcache next = get(0);
            for(size_t y = 0; y < dim; y++)
                for(size_t x = 0; x < dim-width+1; x++)
                    init_cache_by_rectangle(next, y, x, x+width-1);
        }
        for(size_t i = 1; i < width-1; i++) {
            maxcache prev = get(i-1);
            maxcache next = get(i);
            for(size_t y = 0; y < dim-i; y++)
                for(size_t x = 0; x < dim-width+1+i; x++)
                    extend_cache_by_rectangle(prev, next, y, y+i, x, x+width-1-i);
        }
    }

    void init_cache_by_rectangle(maxcache &next, size_t y, size_t x0, size_t x1) {
        next.setmax(y,x0+1, bundles.ell_score(y,y,x0,x1), SW);
        next.setmax(y,x0,   bundles.ell_score(y,y,x1,x0), SE);
    }

    void extend_cache_by_rectangle(const maxcache &prev, maxcache &next,
                                   size_t y0, size_t y1, size_t x0, size_t x1) {
        next.setmax(y0,x0+1, prev.score(y0+1,x0) + bundles.ell_score(y0,y1,x0,x1), NW);
        next.setmax(y0,x0+1, prev.score(y0,  x0) + bundles.ell_score(y1,y0,x0,x1), SW);
        next.setmax(y0,x0,   prev.score(y0+1,x0) + bundles.ell_score(y0,y1,x1,x0), NE);
        next.setmax(y0,x0,   prev.score(y0,  x0) + bundles.ell_score(y1,y0,x1,x0), SE);
    }

    template <typename F>
    void finish_cache_by_rectangle(maxcache &prev, F &setmax,
                                   size_t y0, size_t y1, size_t x) {
        setmax(y0,x, prev.score(y0+1,x) + bundles.ell_score(y0,y1,x,x), NE);
        setmax(y0,x, prev.score(y0,  x) + bundles.ell_score(y1,y0,x,x), SE);
    }
    void inflate_ell(const chimera_lines &chim, vector<vector<size_t>> &emb,
                     size_t &y, size_t &x, size_t h, size_t w, corner c) {
        switch(c) {
            case NW: x--; chim.inflate_bundle(y, y+h, x, x+w, emb); y++; break;
            case SW: x--; chim.inflate_bundle(y+h, y, x, x+w, emb);      break;
            case NE:      chim.inflate_bundle(y, y+h, x+w, x, emb); y++; break;
            case SE:      chim.inflate_bundle(y+h, y, x+w, x, emb);      break;
            default:break;
        }
    }
    
    template <typename F>
    void best_corners(F &setmax, size_t y, size_t x, size_t i) {
        auto cur = get(i);
        size_t h = i+1, w = width-2-i;
        if(x > 0 && x+w-1 < dim && y+h < dim) {
            setmax(y+1,x-1, cur.score(y+1,x)+bundles.ell_score(y, y+h,x-1,x+w-1), NW);
            setmax(y,  x-1, cur.score(y  ,x)+bundles.ell_score(y+h, y,x-1,x+w-1), SW);
        }
        if(x+w < dim && y+h < dim) {
            setmax(y+1,x,   cur.score(y+1,x)+bundles.ell_score(y, y+h,x+w,x),     NE);
            setmax(y,  x,   cur.score(y  ,x)+bundles.ell_score(y+h, y,x+w,x),     SE);
        }
    }

  public:
    void extract_solution(const chimera_lines &chim, vector<vector<size_t>> &emb) {
        size_t bx, by, score=0;
        corner bc;
        auto F = [&bx, &by, &bc, &score](size_t y, size_t x, size_t s, corner c) {
            if(s > score) {
                bx = x; by = y; bc = c; score = s;
            }
        };
        maxcache prev = get(width-2);
        for(size_t y = 0; y < dim-width+1; y++)
            for(size_t x = 0; x < dim; x++)
                finish_cache_by_rectangle(prev, F, y, y+width-1, x);
        if(score == 0) return;
        for(size_t i = width-1; i-- > 0;) {
            inflate_ell(chim, emb, by, bx, i+1, width-2-i, bc);
            bc = static_cast<corner>(get(i).corner(by, bx));
        }
        inflate_ell(chim, emb, by, bx, 0, width-1, bc);
    }

    class solution_generator {
        rectangle_cache *parent;
        size_t dim;
        size_t width;
        vector<vector<std::tuple<size_t, size_t, corner>>> stack;
      public:
        solution_generator(rectangle_cache *p) : 
            parent(p), dim(p->dim), width(p->width), stack(0) {
            stack.emplace_back(0);
            auto &basepoints = stack[0];
            size_t score = 0;
            auto F = [&score, &basepoints](size_t y, size_t x, size_t s, corner c) {
                if(s < score) return;
                else if (s > score) basepoints.clear();
                score = s;
                basepoints.emplace_back(y, x, c);
            };
            maxcache prev = parent->get(width-2);
            for(size_t y = 0; y < dim-width+1; y++)
                for(size_t x = 0; x < dim; x++)
                    parent->finish_cache_by_rectangle(prev, F, y, y+width-1, x);

            while(fill_in_solution());
            print();
        }

        void print() {
            for(size_t i = 0; i < stack.size(); i++) {
                for(std::tuple<size_t, size_t, corner> &z: stack[i]) {
                    switch(std::get<2>(z)) {
                        case NW: std::cout << "NW "; break;
                        case NE: std::cout << "NE "; break;
                        case SW: std::cout << "SW "; break;
                        case SE: std::cout << "SE "; break;
                    }
                } std::cout << std::endl;
            }
        }

      private:
        bool fill_in_solution() {
            if(stack.size() == 0 || stack.size() > width-2) return false;
            size_t y0, x0; corner c0;
            std::tie(y0, x0, c0) = stack.back().back();
            stack.emplace_back(0);
            auto &next = stack.back();
            size_t score = 0;
            auto F = [&next, &score](size_t y, size_t x, size_t s, corner c) {
                if(s < score) return;
                else if(s > score) next.clear();
                score = s;
                next.emplace_back(y, x, c);
            };
            parent->best_corners(F, y0, x0, width-2-stack.size());
            return true;
        }
    };

    solution_generator solutions() {
        return solution_generator(this);
    }
};

void find_clique(chimera_lines chim, bundle_cache &bundles, size_t width,
                         vector<vector<size_t>> &emb) {
    if(width == 1) {
        size_t by, bx, score=0;
        for(size_t y = 0; y < chim.dim; y++)
            for(size_t x = 0; x < chim.dim; x++) {
                size_t s = bundles.ell_score(y,y,x,x);
                if (s > score) {
                    score = s;
                    by = y;
                    bx = x;
                }
            }
        if(score > 0)
            chim.inflate_bundle(by,by,bx,bx,emb);
    } else if(width > 1) {
        auto rects = rectangle_cache(chim.dim, width, bundles);
        rects.solutions();
        rects.extract_solution(chim, emb);
    }
}

void experiment(size_t dim, size_t shore, vector<pair<size_t, size_t>> edges,
                size_t width, vector<vector<size_t>> &emb) {
    chimera_lines chim(dim, shore, edges);
    bundle_cache bundles(chim);
//    chim.verify_bundle_cache(bundles);
    find_clique(chim, bundles, width, emb);
}


}

