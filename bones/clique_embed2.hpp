#pragma once

#include<limits>
#include<utility>
#include<vector>
#include<cstdint>
#include<algorithm>
//#include<assert.h>
#include<iostream>

namespace chimera_clique {

using std::numeric_limits;
using std::vector;
using std::pair;
using std::min;
using std::max;

void Assert(bool thing) {
    if(!thing) throw std::exception();
}

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
        vector<bool> h_mask = h_scan(y0, x0, x1);
        vector<bool> v_mask = v_scan(x0, y0, y1);
        vector<size_t> h_k;
        vector<size_t> v_k;
        for(size_t k = 0; k < shore; k++) {
            if(h_mask[k]) h_k.push_back(k);
            if(v_mask[k]) v_k.push_back(k);
        }
//    std::cout << h_k.size() << "of" << shore << std::endl;
//    std::cout << v_k.size() << "of" << shore << std::endl;
        std::cout << h_k.size() << "/" << v_k.size() << std::endl;
        for(size_t i = 0; i < min(h_k.size(), v_k.size()); i++) {
            chains.emplace_back(0);
            inflate_h(y0, x0, x1, h_k[i], chains.back());
            inflate_v(x0, y0, y1, v_k[i], chains.back());
        }
    }

    template<typename T>
    void verify_bundle_cache(T &bundles) const {
/*        vector<vector<size_t>> ell;
        for(size_t y0 = 0; y0 < dim; y0++)
         for(size_t y1 = 0; y1 < dim; y1++)
          for(size_t x0 = 0; x0 < dim; x0++)
           for(size_t x1 = 0; x1 < dim; x1++) {
            inflate_bundle(y0, y1, x0, x1, ell);
            if(ell.size() != bundles.ell_score(y0, y1, x0, x1)) {
                std::cout << "bad ell size" << y0 << "," << y1 << ","
                          << x0 << "," << x1 << ";" << ell.size() 
                          << "!=" << bundles.ell_score(y0, y1, x0, x1)
                          << "(" << bundles.get_hline_score(y0, min(x0, x1), max(x0, x1))
                          << "," << bundles.get_vline_score(x0, min(y0, y1), max(y0, y1))
                          << ")" << std::endl;

            }
            ell.clear();
           }*/
        for(size_t y = 0; y < dim; y++)
         for(size_t x0 = 0; x0 < dim; x0++)
          for(size_t x1 = x0; x1 < dim; x1++) {
            check_hline(y, x0, x1, bundles);
            check_vline(y, x0, x1, bundles);
          }
    }
       
  protected:
    template<typename T>
    void check_hline(size_t y, size_t x0, size_t x1, T &bundles) const {
        vector<bool> h_mask = h_scan(y, x0, x1);
        size_t s = 0;
        for(size_t k = 0; k < shore; k++) {
            s = s + (h_mask[k]?1:0);
        }
        if (s != bundles.get_hline_score(y, x0, x1)) {
            std::cout << "bad hline size" << y << "," << x0 << "," << x1 << ":" << s << "!=" << bundles.get_hline_score(y, x0, x1) << std::endl;
        }
    }

    template<typename T>
    void check_vline(size_t x, size_t y0, size_t y1, T &bundles) const {
        vector<bool> v_mask = v_scan(x, y0, y1);
        size_t s = 0;
        for(size_t k = 0; k < shore; k++) {
            s = s + (v_mask[k]?1:0);
        }
        if (s != bundles.get_vline_score(x, y0, y1)) {
            std::cout << "bad vline size" << x << "," << y0 << "," << y1 << ":" << s << "!=" << bundles.get_vline_score(x, y0, y1) << std::endl;
        }
    }

    void inflate_h(size_t y, size_t x0, size_t x1, size_t k, vector<size_t> &chain) const {
        if(x0 > x1) std::swap(x0, x1);
        for(size_t x = x0; x <= x1; x++)
            chain.push_back(k + shore*(1 + 2*(x + dim*y)));
        /*
        if(x0 > x1)
            for(size_t x = x0+1; x-- > x1;)
                chain.push_back(k + shore*(1 + 2*(x + dim*y)));
        else
            for(size_t x = x0; x <= x1; x++)
                chain.push_back(k + shore*(1 + 2*(x + dim*y)));
        */
    }

    void inflate_v(size_t x, size_t y0, size_t y1, size_t k, vector<size_t> &chain) const {
        if(y0 > y1) std::swap(y1, y0);
        for(size_t y = y0; y <= y1; y++)
            chain.push_back(k + shore*(0 + 2*(x + dim*y)));
        /*
        if(y0 > y1)
            for(size_t y = y0+1; y-- > y1;)
                chain.push_back(k + shore*(1 + 2*(x + dim*y)));
        else
            for(size_t y = y0; y <= y1; y++)
                chain.push_back(k + shore*(0 + 2*(x + dim*y)));
        */
    }

    vector<bool> h_scan(size_t y, size_t x0, size_t x1) const {
        Assert(y < dim);
        Assert(x0 < dim);
        Assert(x1 < dim);
        vector<bool> K(shore, true);
        if(x0 > x1) std::swap(x0, x1);
        for(size_t k = 0; k < shore; k++)
            K[k] = has_qubit(y, x0, 1, k);
        while(++x0 <= x1) {
            for(size_t k = 0; k < shore; k++)
                K[k] = K[k] & has_qubit(y, x0, 1, k) & has_ext(y, x0, 1, k);
        }
        return K;
    }

    vector<bool> v_scan(size_t x, size_t y0, size_t y1) const {
        Assert(x < dim);
        Assert(y0 < dim);
        Assert(y1 < dim);
        vector<bool> K(shore, true);
        if(y0 > y1) std::swap(y0, y1);
        for(size_t k = 0; k < shore; k++)
            K[k] = has_qubit(y0, x, 0, k);
        while(++y0 <= y1) {
            for(size_t k = 0; k < shore; k++)
                K[k] = K[k] & has_qubit(y0, x, 0, k) & has_ext(y0, x, 0, k);
        }
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
//            delete [] line_score;
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

    size_t get_hline_score(size_t y, size_t x0, size_t x1) const {
        Assert(y < dim);
        Assert(x0 <= x1);
        Assert(x1 < dim);
        return line_score[orthstride + y*linestride + (x1*x1+x1)/2 + x0];
    }

    size_t get_vline_score(size_t x, size_t y0, size_t y1) const {
        
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
                    memset(&scanline, 0, sizeof(size_t)*(z+1));
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
                                shoreline[k] = null_node; // ded
                            }
                        }
                    }
                    size_t offset = u*orthstride + w*linestride + (z*z+z)/2;
                    for(size_t z0 = 0; z0 <= z; z0++)
                        line_score[offset + z0] = scanline[z0];
                }
            }
        }
    }    
};


class rectangle_cache {
  public:
    class maxcache {
      public:
        const size_t rows;
        const size_t cols;
      private:
        size_t *mem;
      public:
        maxcache(size_t r, size_t c, size_t *m) : rows(r), cols(c), mem(m) {};
        void setmax(size_t y, size_t x, size_t m) { 
            Assert(y < rows);
            Assert(x < cols);
            mem[y*cols+x] = max(mem[y*cols+x], m);
        }
        size_t get(size_t y, size_t x) const { 
//            std::cout << y << "<?" << rows << "   " << x << "<?" << cols << std::endl;
            Assert(y < rows);
            Assert(x < cols);
            return mem[y*cols + x];
        }
        size_t advance(size_t y, size_t x, size_t h, size_t w,
                       size_t dy, size_t dx,
                       const bundle_cache &bundles) const {
            size_t cur = get(y+dy, x)/4;
            switch(dy + 2*dx) {
/*                case 0: return (cur + bundles.ell_score(y,y+h-1,x,x+w-1))*4 + 0;
                case 1: return (cur + bundles.ell_score(y+h-1,y,x+w-1,x))*4 + 1;
                case 2: return (cur + bundles.ell_score(y,y+h-1,x,x+w-1))*4 + 2;
                case 3: return (cur + bundles.ell_score(y+h-1,y,x+w-1,x))*4 + 3;*/
                case 0: return (cur + bundles.ell_score(y+h-1,y,x,x+w-1))*4 + 0;
                case 1: return (cur + bundles.ell_score(y,y+h-1,x,x+w-1))*4 + 1;
                case 2: return (cur + bundles.ell_score(y+h-1,y,x+w-1,x))*4 + 2;
                case 3: return (cur + bundles.ell_score(y,y+h-1,x+w-1,x))*4 + 3;
                default: throw std::exception();
            }
        }
        void retreat(size_t &y, size_t &x, size_t h, size_t w,
                       const chimera_lines &chim,
                       const bundle_cache &bundles,
                       vector<vector<size_t>> &emb) {
            std::cout << "retreating " << y << "o" << rows << "," << x << "o" << cols << std::endl;
            size_t move = get(y, x)&3;
            std::cout << "(" << move << ")!" << std::endl;
            size_t yh = y+h-1, xw = x+w-1;
            switch(move) {
                case 0: //chim.inflate_bundle(yh,y,x,xw, emb);
                    break;
                case 1: //chim.inflate_bundle(y,yh,xw,x, emb);
                    y++;      
                    break;
                case 2: //chim.inflate_bundle(yh,y,xw,x, emb);
                    x--; 
                    break;
                case 3: chim.inflate_bundle(y,yh,x,xw, emb);
                    y++; x--; 
                    break;
                default: throw std::exception();
            }
        }
    };
    
  private:
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
    rectangle_cache(size_t d, size_t w) : 
            dim(d),
            width(w),
            mem(new size_t[memsize()]{}) {
        mem[0] = width;
        for(size_t i = 1; i < width-1; i++)
            mem[i] = mem[i-1] + memsize(i-1);
    }

    ~rectangle_cache() {
        if (mem != nullptr) {
//            delete [] mem;
            mem = nullptr;
        }
    }

    maxcache get(size_t i) {
        Assert(i < width-1);
        return maxcache(dim-i, dim-width+i+2, mem + mem[i]);
    }

    size_t rectangle_width(size_t i) {
        return width-1-i;
    }
    size_t rectangle_height(size_t i) {
        return i+1;
    }

    void print() {
        for(size_t i = 0; i < width-1; i++) {
            auto m = get(i);
            std::cout << mem[i] << ':' << memsize(i) << "?"<< std::endl;
            for(size_t y = 0; y < m.rows; y++) {
                for(size_t x = 0; x < m.cols; x++) {
                    std::cout << m.get(y, x)/4 << '~' << (m.get(y, x)&3) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
};

void init_cache(size_t dim, const bundle_cache &bundles, size_t width,
                rectangle_cache &rects) {
    size_t w = width;
    auto next = rects.get(0);
    Assert(next.rows == dim);
    Assert(next.cols == dim-w+2);
    Assert(rects.rectangle_width(0) == w-1);
    Assert(rects.rectangle_height(0) == 1);

    for(size_t y = 0; y < dim; y++) {
        for(size_t x = 0; x < dim-w+1; x++) {
            next.setmax(y, x, bundles.ell_score(y, y, x, x+w-1)*4 + 2);
            next.setmax(y, x+1, bundles.ell_score(y, y, x+w-1, x)*4);
        }
    }
}

void extend_cache(size_t dim, const bundle_cache &bundles, size_t width,
                  rectangle_cache &rects, size_t i) {
    size_t h = i + 1;
    size_t w = width-h;
    auto prev = rects.get(i);
    auto next = rects.get(i+1);
    Assert(rects.rectangle_width(i) == w);
    Assert(rects.rectangle_height(i) == h);
    Assert(rects.rectangle_width(i+1) == w-1);
    Assert(rects.rectangle_height(i+1) == h+1);


    for (size_t y = 0; y < dim-h-1; y++) {
        for(size_t x = 0; x < dim-w+1; x++) {
            next.setmax(y, x+1, prev.get(y+1, x) +
                                bundles.ell_score(y, y+h-1, x, x+w-1));
            next.setmax(y, x+1, prev.get(y, x) +
                                bundles.ell_score(y+h-1, y, x, x+w-1));
            next.setmax(y, x, prev.get(y+1, x) +
                              bundles.ell_score(y, y+h-1, x+w-1, x));
            next.setmax(y, x, prev.get(y, x) +
                              bundles.ell_score(y+h-1, y, x+w-1, x));


//            next.setmax(y, x+1, prev.advance(y, x, h, w, 1, 0, bundles));
//            next.setmax(y, x, prev.advance(y, x, h, w, 1, 1, bundles));
//            next.setmax(y+1, x+1, prev.advance(y, x, h, w, 0, 0, bundles));
//            next.setmax(y+1, x, prev.advance(y, x, h, w, 0, 1, bundles));
//            next.setmax(y, x+1, max(prev.advance(y, x, h, w, 1, 1, bundles),
//                                    prev.advance(y, x, h, w, 0, 1, bundles)));
//            next.setmax(y, x, max(prev.advance(y, x, h, w, 1, 0, bundles),
//                                  prev.advance(y, x, h, w, 0, 0, bundles)));
        }
    }
}

rectangle_cache clique_cache(chimera_lines &chim, bundle_cache &bundles,
                             size_t width) {

    rectangle_cache rects(chim.dim, width);

    init_cache(chim.dim, bundles, width, rects);
    for (size_t i = 0; i < width-2; i++)
        extend_cache(chim.dim, bundles, width, rects, i);

    rects.print();

    return rects;
}


bool kickoff_solution(chimera_lines &chim, bundle_cache &bundles,
                      rectangle_cache &rects, int width, size_t &by,
                      size_t &bx, vector<vector<size_t>> &emb) {
    size_t score = 0;
    bool up = false;
    auto prev = rects.get(width-2);
    size_t h = width-1;
    size_t w = 1;
    Assert(rects.rectangle_width(width-2) == w);
    Assert(rects.rectangle_height(width-2) == h);

    for(size_t y = 0; y < chim.dim-h; y++) {
        for(size_t x = 0; x < chim.dim; x++) {
            size_t s = prev.get(y, x)/4 + bundles.ell_score(y, y+h, x, x);
            if(s > score) { score = s; up = true; by = y; bx = x; }

            s = prev.get(y+1, x)/4 + bundles.ell_score(y+h, y, x, x);
            if(s > score) { score = s; up = false; by = y; bx = x; }

            size_t a; size_t b;
            if((a = prev.get(y, x)/4 + bundles.ell_score(y, y+h, x, x)) > 
               (b = prev.get(y+1, x)/4 + bundles.ell_score(y+h, y, x, x))) {
                std::cout << a << "~" << 1 << " ";
            } else {
                std::cout << b << "~" << 0 << " ";
            }
        }
        std::cout << std::endl;
    }
    if(score == 0) return false;
    if(up) {
        std::cout << "up" << std::endl;
        chim.inflate_bundle(by+h, by, bx, bx, emb);
    } else {
        chim.inflate_bundle(by, by+h, bx, bx, emb);
        by++;
    }
    for(auto &chain : emb) {
        for(auto &q: chain) std::cout << q << ">";
        std::cout << std::endl;
    }
    return true;
}


void extract_solution(chimera_lines &chim, bundle_cache &bundles,
                      rectangle_cache &rects, size_t width, size_t i,
                      size_t &y, size_t &x, vector<vector<size_t>> &emb) {
    auto prev = rects.get(width-2-i);
    auto next = rects.get(width-3-i);
    size_t ph = width-1-i; Assert(ph == rects.rectangle_height(width-2-i));
    size_t pw = i+1; Assert(pw == rects.rectangle_width(width-2-i));
    size_t nh = ph-1; Assert(nh == rects.rectangle_height(width-3-i));
    size_t nw = pw+1; Assert(nw == rects.rectangle_width(width-3-i));
    size_t Lh = ph;
    size_t Lw = nw;
    size_t cur = prev.get(y, x);
    if(x > 0) {
        if(y+1 < next.rows && x+Lw < chim.dim &&
           cur == next.get(y+1, x-1) + bundles.ell_score(y, y+Lh-1, x, x+Lw-1)) {
            chim.inflate_bundle(y, y+Lh-1, x, x+Lw-1, emb);
            y++; x--;
            return;
        }
        if(y < next.rows && x+Lw < chim.dim &&
           cur == next.get(y, x-1) + bundles.ell_score(y, y+Lh-1, x, x+Lw-1)) {
            chim.inflate_bundle(y+Lh-1, y, x, x+Lw-1, emb);
            x--;
            return;
        }
    }
    if(y+1 < next.rows && x < next.cols &&
       cur == next.get(y+1, x) + bundles.ell_score(y, y+Lh-1, x+Lw-1, x)) {
        chim.inflate_bundle(y, y+Lh-1, x+Lw-1, x, emb);
        y++;
        return;
    }
    if(y < next.rows && x < next.cols &&
       cur == next.get(y, x) + bundles.ell_score(y, y+Lh-1, x+Lw-1, x)) {
        chim.inflate_bundle(y+Lh-1, y, x+Lw-1, x, emb);
        return;
    }
    Assert(false);
}

vector<vector<size_t>> native_clique_embed(chimera_lines chim,
                                        bundle_cache bundles,
                                        size_t width) {

    for(size_t y = 0; y < chim.dim; y++) {
        for(size_t x = 0; x < chim.dim; x++)
            std::cout << bundles.ell_score(y,y,x,x) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    vector<vector<size_t>> emb;
    if(width == 0) {
    } else if(width == 1) {
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
    } else {
        auto rects = clique_cache(chim, bundles, width);
        size_t by, bx;

        if(kickoff_solution(chim, bundles, rects, width, by, bx, emb))
            for(size_t i=0; i < width-1; i++) {
                auto prev = rects.get(width-i-2);
//                prev.retreat(by, bx, width-i-1, i+2, chim, bundles, emb);
            }
    }
    return emb;
}

vector<vector<size_t>> experiment(size_t dim, size_t shore,
                                  vector<pair<size_t, size_t>> edges,
                                  size_t width) {
    chimera_lines chim(dim, shore, edges);
    bundle_cache bundles(chim);
    chim.verify_bundle_cache(bundles);
    vector<vector<size_t>> r;
    r = native_clique_embed(chim, bundles, width);
    return r;
}

vector<vector<size_t>> experiment2(size_t dim, size_t shore,
                                  vector<pair<size_t, size_t>> edges,
                                  size_t width) {
    chimera_lines chim(dim, shore, edges);
    bundle_cache bundles(chim);
    return native_clique_embed(chim, bundles, width);
}


}

