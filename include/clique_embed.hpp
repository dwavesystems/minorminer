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
    if(!thing) throw std::exception();
}

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

const size_t null_node = numeric_limits<size_t>::max();

class chimera_lines {
  public:
    const size_t dim[2];
    const size_t shore;
  private:
    vector<bool> nodemask;
    vector<bool> edgemask;

  public:
    chimera_lines(size_t dy, size_t dx, size_t s, vector<size_t> nodes,
                  vector<pair<size_t, size_t>> edges) :
                  dim{dy, dx}, shore(s), nodemask(dy*dx*s*2, false),
                  edgemask(dy*dx*s*2, false) {
        if(shore > 256)
            throw std::exception();
        for(auto &q: nodes) nodemask[q] = true;
        size_t stride_0 = dim[1]*shore*2;
        size_t stride_1 = shore*2;
        for(auto &pq: edges) {
            size_t root = ext_root(pq.first, pq.second, stride_0, stride_1);
            if (root != null_node)
                edgemask[root] = true;
        }
    }

    chimera_lines(size_t dy, size_t dx, size_t s,
                  vector<pair<size_t, size_t>> edges) : dim{dy, dx},
                  shore(s), nodemask(dy*dx*s*2, false),
                  edgemask(dy*dx*s*2, false) {
        size_t stride_0 = dim[1]*shore*2;
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
        Assert(y < dim[0]);
        Assert(x < dim[1]);
        Assert(u < 2);
        Assert(k < shore);
        return nodemask[k + shore*(u + 2*(x + dim[1]*y))];
    }

    //! returns true if the edge (y, x, u, k) ~ (y-1+u, x-u, u, k)
    //! is contained within this graph
    bool has_ext (size_t y, size_t x, size_t u, size_t k) const {
        Assert(y < dim[0]);
        Assert(x < dim[1]);
        Assert(u < 2);
        Assert(k < shore);
        return edgemask[k + shore*(u + 2*(x + dim[1]*y))];
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
    void inflate_ell(size_t y0, size_t y1, size_t x0, size_t x1,
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
            chain.push_back(k + shore*(u + 2*(x + dim[1]*y)));
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
    const size_t dim[2];
    const size_t linestride[2];
    const size_t orthstride;
    uint8_t *line_score;
    bundle_cache(const bundle_cache&) = delete;
    bundle_cache(bundle_cache &&) = delete;

  public:
    ~bundle_cache() {
        if (line_score != nullptr) {
            delete [] line_score;
            line_score = nullptr;
        }
    }
    bundle_cache(const chimera_lines &chim) :
                 dim{chim.dim[0], chim.dim[1]},
                 linestride{(dim[0]*dim[0]+dim[0])/2, (dim[1]*dim[1]+dim[1])/2},
                 orthstride(dim[1]*linestride[0]),
                 line_score(new uint8_t[orthstride + dim[0]*linestride[1]]{}) {
        compute_line_scores(chim);
    }

    size_t ell_score(size_t y0, size_t y1, size_t x0, size_t x1) const {
        return min(get_hline_score(y0, min(x0, x1), max(x0, x1)),
                   get_vline_score(x0, min(y0, y1), max(y0, y1)));
    }

    inline size_t get_hline_score(size_t y, size_t x0, size_t x1) const {
        Assert(y < dim[0]);
        Assert(x0 <= x1);
        Assert(x1 < dim[1]);
        return line_score[orthstride + y*linestride[0] + (x1*x1+x1)/2 + x0];
    }

    inline size_t get_vline_score(size_t x, size_t y0, size_t y1) const {
        Assert(x < dim[1]);
        Assert(y0 <= y1);
        Assert(y1 < dim[0]);
        return line_score[x*linestride[1] + (y1*y1+y1)/2 + y0];
    }

  private:
    void compute_line_scores(const chimera_lines &chim) {
        size_t shoreline[chim.shore];
        size_t scanline[max(chim.dim[0], chim.dim[1])];
        for(size_t u = 0; u < 2; u++) {
            size_t p[2];
            size_t &y = p[0]; size_t &z = p[u];
            size_t &x = p[1]; size_t &w = p[1-u];
            for (w = 0; w < chim.dim[1-u]; w++) {
                for (z = 0; z < chim.dim[u]; z++) {
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
                    size_t offset = u*orthstride + w*linestride[u] + (z*z+z)/2;
                    std::copy(scanline, scanline+z+1, line_score+offset);
                }
            }
        }
    }    
};


class clique_cache {
    friend class clique_iterator;  
  public:
    //pesky double-frees
    clique_cache(const clique_cache&) = delete; 
    clique_cache(clique_cache &&) = delete;
    
    class maxcache {
      public:
        const size_t rows;
        const size_t cols;
      private:
        size_t *mem;
      public:
        maxcache(size_t r, size_t c, size_t *m) : rows(r), cols(c), mem(m) {};
        void setmax(size_t y, size_t x, size_t s, corner c) { 
            Assert(y < rows);
            Assert(x < cols);
            size_t old_s = score(y, x);
            if(s == old_s) {
                mem[y*cols+x] |= c;
            } else if(s > old_s) {
                mem[y*cols+x] = (s << corner::shift) | c;
            }
        }
        size_t score(size_t y, size_t x) const {
            Assert(y < rows);
            Assert(x < cols);
            return mem[y*cols + x] >> corner::shift;
        }
        corner corners(size_t y, size_t x) const {
            Assert(y < rows);
            Assert(x < cols);
            return static_cast<corner>(mem[y*cols + x] & corner::mask);
        }
    };
    
  private:
    chimera_lines &chim;
    bundle_cache &bundles;
    size_t dim[2];
    size_t width;
    size_t memsize(size_t i) const {
        return (dim[0]-i)*(dim[1]-width+i+2);
    }

    size_t *mem;
    size_t memsize() const {
        size_t size = 0;
        for(size_t i = 0; i < width-1; i++)
            size += memsize(i);
        return size + width;
    }

  public:
    clique_cache(chimera_lines &c, bundle_cache &b, size_t w) : 
            chim(c),
            bundles(b),
            dim{chim.dim[0], chim.dim[1]},
            width(w),
            mem(new size_t[memsize()]{}) {
        mem[0] = width;
        for(size_t i = 1; i < width-1; i++)
            mem[i] = mem[i-1] + memsize(i-1);
        compute_cache();
    }

    ~clique_cache() {
        if (mem != nullptr) {
            delete [] mem;
            mem = nullptr;
        }
    }

    maxcache get(size_t i) {
        Assert(i < width-1);
        return maxcache(dim[0]-i, dim[1]-width+i+2, mem + mem[i]);
    }

    void print() {
        for(size_t i = 0; i < width-1; i++) {
            maxcache m = get(i);
            std::cout << mem[i] << ':' << memsize(i) << "?"<< std::endl;
            for(size_t y = 0; y < m.rows; y++) {
                for(size_t x = 0; x < m.cols; x++) {
                    std::cout << m.score(y, x) << '~' << (m.corners(y, x)) << " ";
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
            for(size_t y = 0; y < dim[0]; y++)
                for(size_t x = 0; x < dim[1]-width+1; x++)
                    init_cache_by_rectangle(next, y, x, x+width-1);
        }
        for(size_t i = 1; i < width-1; i++) {
            maxcache prev = get(i-1);
            maxcache next = get(i);
            for(size_t y = 0; y < dim[0]-i; y++)
                for(size_t x = 0; x < dim[1]-width+1+i; x++)
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
    void inflate_ell(vector<vector<size_t>> &emb,
                     size_t &y, size_t &x, size_t h, size_t w, corner c) {
        switch(c) {
            case NW: x--; chim.inflate_ell(y, y+h, x, x+w, emb); y++; break;
            case SW: x--; chim.inflate_ell(y+h, y, x, x+w, emb);      break;
            case NE:      chim.inflate_ell(y, y+h, x+w, x, emb); y++; break;
            case SE:      chim.inflate_ell(y+h, y, x+w, x, emb);      break;
            default:break;
        }
    }
    corner inflate_first_ell(vector<vector<size_t>> &emb,
                     size_t &y, size_t &x, size_t h, size_t w, corner c) {
        if(NW&c) { x--; chim.inflate_ell(y, y+h, x, x+w, emb); y++; return NW; }
        if(SW&c) { x--; chim.inflate_ell(y+h, y, x, x+w, emb);      return SW; }
        if(NE&c) {      chim.inflate_ell(y, y+h, x+w, x, emb); y++; return NE; }
        if(SE&c) {      chim.inflate_ell(y+h, y, x+w, x, emb);      return SE; }
        throw std::exception();
    }


  public:
    void extract_solution(vector<vector<size_t>> &emb) {
        size_t bx, by, score=0;
        corner bc;
        auto F = [&bx, &by, &bc, &score](size_t y, size_t x, size_t s, corner c) {
            if(s > score) {
                bx = x; by = y; bc = c; score = s;
            }
        };
        maxcache prev = get(width-2);
        for(size_t y = 0; y < dim[0]-width+1; y++)
            for(size_t x = 0; x < dim[1]; x++)
                finish_cache_by_rectangle(prev, F, y, y+width-1, x);
        if(score == 0) return;
        for(size_t i = width-1; i-- > 0;) {
            inflate_first_ell(emb, by, bx, i+1, width-2-i, bc);
            bc = static_cast<corner>(get(i).corners(by, bx));
        }
        inflate_first_ell(emb, by, bx, 0, width-1, bc);
    }
};

class clique_iterator {
    chimera_lines &chim;
    clique_cache &cliq;
    size_t dim[2];
    size_t width;
    vector<std::tuple<size_t, size_t, corner>> basepoints;
    vector<std::tuple<size_t, size_t, size_t, corner>> stack;
    vector<vector<size_t>> emb;
    
  public:
    clique_iterator(chimera_lines &c, clique_cache &q) :
                    chim(c), cliq(q), dim{chim.dim[0], chim.dim[1]},
                    width(cliq.width), basepoints(0), stack(0) {

        //to prepare the iterator, we first compute the list of optimal 
        //basepoints -- which consist of a (y, x) pair to denote the rectangle
        //location, and a corner c to denote the orientation of the ell.  
        size_t score = 0;
        auto F = [=, &score](size_t y, size_t x, size_t s, corner c) {
            if(s < score) return;
            else if (s > score) basepoints.clear();
            score = s;
            basepoints.emplace_back(y, x, c);
        };
        clique_cache::maxcache prev = cliq.get(width-2);
        for(size_t y = 0; y < dim[0]-width+1; y++)
            for(size_t x = 0; x < dim[1]; x++)
                cliq.finish_cache_by_rectangle(prev, F, y, y+width-1, x);
    }

  private:
    bool advance() {
        //first, peel back the zeros (exhausted solutions) until we hit a
        //nonzero corner
        size_t n, by, bx;
        corner bc;
        while(stack.size()) {
            std::tie(n, by, bx, bc) = stack.back();                
            if(bc) {
                while(emb.size() > n)
                    emb.pop_back();
                break;
            } else {
                stack.pop_back();
            }
        }
        //now, regrow the stack -- if we peeled back the entire stack, we
        //didn't trim the embedding above and we throw the whole thing out
        if(stack.size() == 0) emb.clear();
        while(grow_stack());
        return (emb.size() != 0);
    }

    bool grow_stack() {
        //this grows a stack one step at a time.  the stack is inhomogeneous, so
        //it's a little bothersome to do so -- thus, this isn't a nice recursive
        //function -- we do one step at a time and return true if there's still
        //more work to do; false otherwise.

        size_t n, by, bx, i;
        corner lc, bc;
        if(stack.size() == 0) {
            //in this case, we've used up the last basepoint that was examined
            //go to the next basepoint (if available) and prime the stack and
            //the current embedding
            if(basepoints.size() == 0)
                return false;
            std::tie(by, bx, bc) = basepoints.back();
            basepoints.pop_back();
            cliq.inflate_first_ell(emb, by, bx, width-1, 0, bc);
            bc = cliq.get(width-2).corners(by, bx);
            //we'll grow the stack from here using the ell located at by, bx
            //with any of the corners in bc 
            stack.emplace_back(emb.size(), by, bx, bc);
            return true;
        } else if((i = stack.size()) < width) {
            //in this case, we're working on a nonempty stack -- so we grow the
            //clique from here
            std::tie(n, by, bx, bc) = stack.back();
            lc = cliq.inflate_first_ell(emb, by, bx, width-i-1, i, bc);
            //here we grow the stack with the first corner in bc -- mark it off
            //as used, so that the next time the stack reaches this depth we
            //won't repeat it
            std::get<3>(stack[i-1]) = static_cast<corner>(lc^bc);
            if(i < width-1) {
                bc = cliq.get(width-2-i).corners(by, bx);
                stack.emplace_back(emb.size(), by, bx, bc);
                return true;
            }
        }
        return false;
    }
  public:
    bool next(vector<vector<size_t>> &e) {
        e.clear();
        if(advance()) {
            for(auto &chain: emb)
                e.emplace_back(chain);
            return true;
        } else return false;
    }
};

void find_clique(chimera_lines &chim, bundle_cache &bundles, size_t width,
                         vector<vector<size_t>> &emb) {
    if(width == 1) {
        size_t by, bx, score=0;
        for(size_t y = 0; y < chim.dim[0]; y++)
            for(size_t x = 0; x < chim.dim[1]; x++) {
                size_t s = bundles.ell_score(y,y,x,x);
                if (s > score) {
                    score = s;
                    by = y;
                    bx = x;
                }
            }
        if(score > 0)
            chim.inflate_ell(by,by,bx,bx,emb);
    } else if(width > 1) {
        clique_cache rects(chim, bundles, width);
        clique_iterator(chim, rects).next(emb);
    }
}


void experiment(size_t dim0, size_t dim1, size_t shore,
                vector<pair<size_t, size_t>> edges,
                size_t width, vector<vector<size_t>> &emb) {
    chimera_lines chim(dim0, dim1, shore, edges);
    bundle_cache bundles(chim);
//    chim.verify_bundle_cache(bundles);
    find_clique(chim, bundles, width, emb);
}

class clique_generator {
    chimera_lines chim;
    bundle_cache bundles;
    clique_cache rects;
    clique_iterator iter;
  public:
    clique_generator(size_t dim0, size_t dim1, size_t shore,
                     vector<pair<size_t, size_t>> &edges, size_t width) :
        chim(dim0, dim1, shore, edges),
        bundles(chim),
        rects(chim, bundles, width),
        iter(chim, rects) {}

    bool next(vector<vector<size_t>> &emb) {
        return iter.next(emb);
    }
};


}

