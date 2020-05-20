#pragma once
#include "util.hpp"

namespace busclique {

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

template<typename topo_spec> class clique_iterator;

template<typename topo_spec>
class clique_cache {
    friend class clique_iterator<topo_spec>;  
  public:
    //prevent double-frees by forbidding moving & copying
    clique_cache(const clique_cache&) = delete; 
    clique_cache(clique_cache &&) = delete;
    
  private:
    const cell_cache<topo_spec> &cells;
    const bundle_cache<topo_spec> &bundles;
    const size_t width;
    size_t *mem;

    size_t memrows(size_t i) const {
        if (i < width) return cells.topo.dim[0]-i;
        else if (i == width) return 1;
        else throw "memrows";
    }
    size_t memcols(size_t i) const {
        if (i < width-1) return cells.topo.dim[1]-width+i+2;
        else if (i == width-1) return cells.topo.dim[1];
        else throw "memcols";
    }

    size_t memsize(size_t i) const {
        return memrows(i)*memcols(i);
    }

    size_t memsize() const {
        size_t size = 0;
        for(size_t i = 0; i < width; i++)
            size += memsize(i) + 1;
        return size;
    }

  public:
    template<typename C>
    clique_cache(const cell_cache<topo_spec> &c, const bundle_cache<topo_spec> &b, size_t w, C &check) : 
            cells(c),
            bundles(b),
            width(w),
            mem(new size_t[memsize()]{}) {
        mem[0] = width;
        for(size_t i = 1; i < width; i++)
            mem[i] = mem[i-1] + memsize(i-1);
        compute_cache(check);
    }

    ~clique_cache() {
        if (mem != nullptr) {
            delete [] mem;
            mem = nullptr;
        }
    }

    maxcache get(size_t i) const {
        Assert(i < width);
        return maxcache(memrows(i), memcols(i), mem + mem[i]);
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
    
    template<typename C>
    void compute_cache(C &check) {
        {
            maxcache next = get(0);
            for(size_t y = 0; y < cells.topo.dim[0]; y++)
                for(size_t x = 0; x < cells.topo.dim[1]-width+1; x++)
                    init_cache_by_rectangle(next, y, x, x+width-1, check);
        }
        for(size_t i = 1; i < width-1; i++) {
            maxcache prev = get(i-1);
            maxcache next = get(i);
            for(size_t y = 0; y < cells.topo.dim[0]-i; y++)
                for(size_t x = 0; x < cells.topo.dim[1]-width+1+i; x++)
                    extend_cache_by_rectangle(prev, next, y, y+i, x, x+width-1-i,
                                              check);
        }
        {
            maxcache prev = get(width-2);
            maxcache next = get(width-1);
            for(size_t y = 0; y < next.rows; y++)
                for(size_t x = 0; x < next.cols; x++)
                    finish_cache_by_rectangle(prev, next, y, y+width-1, x, check);
        }
    }

    template<typename C>
    void init_cache_by_rectangle(maxcache &next, size_t y, size_t x0, size_t x1, 
                                 C& check) {
        if(check(y,x0,y,y,x0,x1))
            next.setmax(y,x0+1, bundles.score(y,x0,y,y,x0,x1), SW);
        if(check(y,x1,y,y,x0,x1))
            next.setmax(y,x0,   bundles.score(y,x1,y,y,x0,x1), SE);
    }

    template<typename C>
    void extend_cache_by_rectangle(const maxcache &prev, maxcache &next,
                                   size_t y0, size_t y1, size_t x0, size_t x1,
                                   C& check) {
        if(check(y0,x0,y0,y1,x0,x1))
            next.setmax(y0,x0+1, 
                        prev.score(y0+1,x0) + bundles.score(y0,x0,y0,y1,x0,x1), NW);
        if(check(y1,x0,y0,y1,x0,x1))
            next.setmax(y0,x0+1,
                        prev.score(y0,  x0) + bundles.score(y1,x0,y0,y1,x0,x1), SW);
        if(check(y0,x1,y0,y1,x0,x1))
            next.setmax(y0,x0,
                        prev.score(y0+1,x0) + bundles.score(y0,x1,y0,y1,x0,x1), NE);
        if(check(y1,x1,y0,y1,x0,x1))
            next.setmax(y0,x0,
                        prev.score(y0,  x0) + bundles.score(y1,x1,y0,y1,x0,x1), SE);
    }

    template <typename C>
    void finish_cache_by_rectangle(const maxcache &prev, maxcache &next,
                                   size_t y0, size_t y1, size_t x, C& check) const {
        if(check(y0,x,y0,y1,x,x))
            next.setmax(y0,x, prev.score(y0+1,x) + bundles.score(y0,x,y0,y1,x,x), NE);
        if(check(y1,x,y0,y1,x,x))
            next.setmax(y0,x, prev.score(y0,  x) + bundles.score(y1,x,y0,y1,x,x), SE);
    }
    void inflate_ell(vector<vector<size_t>> &emb,
                     size_t &y, size_t &x, size_t h, size_t w, corner c) const {
        switch(c) {
            case NW: x--; bundles.inflate(y,  x,  y,y+h,x,x+w, emb); y++; break;
            case SW: x--; bundles.inflate(y+h,x,  y,y+h,x,x+w, emb);      break;
            case NE:      bundles.inflate(y,  x+w,y,y+h,x,x+w, emb); y++; break;
            case SE:      bundles.inflate(y+h,x+w,y,y+h,x,x+w, emb);      break;
            default: throw std::exception();
        }
    }
    corner inflate_first_ell(vector<vector<size_t>> &emb,
                     size_t &y, size_t &x, size_t h, size_t w, corner c) const {
        corner c0 = static_cast<corner>(1<<first_bit[c]);
        inflate_ell(emb, y, x, h, w, c0);
        return c0;
    }


  public:
    bool extract_solution(vector<vector<size_t>> &emb) {
        size_t bx, by, bscore=0;
        maxcache scores = get(width-1);
        for(size_t y = 0; y < cells.topo.dim[0]-width+1; y++)
            for(size_t x = 0; x < cells.topo.dim[1]; x++) {
                size_t s = scores.score(y, x);
                if (bscore < s) { 
                    bx = x; by = y; bscore = s;
                }
            }
        if(bscore == 0) return false;
        corner bc = static_cast<corner>(scores.corners(by, bx));
        for(size_t i = width-1; i-- > 0;) {
            inflate_first_ell(emb, by, bx, i+1, width-2-i, bc);
            bc = static_cast<corner>(get(i).corners(by, bx));
        }
        inflate_first_ell(emb, by, bx, 0, width-1, bc);
        return true;
    }
};

template<typename topo_spec>
class clique_iterator {
    const cell_cache<topo_spec> &cells;
    const clique_cache<topo_spec> &cliq;
    size_t width;
    vector<std::tuple<size_t, size_t, corner>> basepoints;
    vector<std::tuple<size_t, size_t, size_t, corner>> stack;
    vector<vector<size_t>> emb;
    
  public:
    clique_iterator(const cell_cache<topo_spec> &c,
                    const clique_cache<topo_spec> &q) :
                    cells(c), cliq(q),
                    width(cliq.width), basepoints(0), stack(0) {

        //to prepare the iterator, we first compute the list of optimal 
        //basepoints -- which consist of a (y, x) pair to denote the rectangle
        //location, and a corner c to denote the orientation of the ell.  
        size_t score = 0;
        maxcache scores = cliq.get(width-1);
        for(size_t y = 0; y < scores.rows; y++)
            for(size_t x = 0; x < scores.cols; x++) {
                size_t s = scores.score(y, x);
                if(s < score) continue;
                else if (s > score) basepoints.clear();
                score = s;
                basepoints.emplace_back(y, x, scores.corners(y, x));
            }
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

}
