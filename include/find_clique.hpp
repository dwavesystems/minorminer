#include "util.hpp"
#include "cell_cache.hpp"
#include "bundle_cache.hpp"
#include "clique_cache.hpp"
#include "small_cliques.hpp"

namespace busclique {


template<typename T>
size_t get_maxlen(vector<T> &emb, size_t size) {
    auto cmp = [](T a, T b) { return a.size() < b.size(); };
    std::sort(emb.begin(), emb.end(), cmp);
    return emb[size-1].size();
}

template<typename topo_spec>
bool find_clique_nice(const topo_spec &,
                      const vector<size_t> &nodes,
                      const vector<pair<size_t, size_t>> &edges,
                      size_t size,
                      vector<vector<size_t>> &emb,
                      size_t max_length = 0);

template<>
bool find_clique_nice(const chimera_spec &topo, 
                      const vector<size_t> &nodes,
                      const vector<pair<size_t, size_t>> &edges,
                      size_t size,
                      vector<vector<size_t>> &emb,
                      size_t) {
    switch(size) {
      case 0: return true;
      case 1: return find_generic_1(nodes, emb);
      case 2: return find_generic_2(edges, emb);
      default: break;
    }
    cell_cache<chimera_spec> cells(topo, nodes, edges);
    bundle_cache<chimera_spec> bundles(cells);

    size_t shore = cells.shore;
    if(size <= shore) {
        size_t by, bx, score=0;
        for(size_t y = 0; y < cells.dim[0]; y++)
            for(size_t x = 0; x < cells.dim[1]; x++) {
                size_t s = bundles.score(y,x,y,y,x,x);
                if (s > score) {
                    score = s;
                    by = y;
                    bx = x;
                }
            }
        if(score >= size) {
            bundles.inflate(by,bx,by,by,bx,bx,emb);
            return true;
        }
    }
    size_t minw = (size + shore - 1)/shore;
    size_t maxw = min(cells.dim[0], cells.dim[1]);
    for(size_t width = minw; width <= maxw; width++) {
        clique_cache<chimera_spec> rects(cells, bundles, width);
        clique_iterator<chimera_spec> iter(cells, rects);
        if(iter.next(emb)) {
            if(emb.size() < size) emb.clear();
            else return true;
        }
    }
    return false;
}

template<>
bool find_clique_nice(const pegasus_spec &topo, 
                      const vector<size_t> &nodes,
                      const vector<pair<size_t, size_t>> &edges,
                      size_t size,
                      vector<vector<size_t>> &emb,
                      size_t max_length) {

    switch(size) {
      case 0: return true;
      case 1: return find_generic_1(nodes, emb);
      case 2: return find_generic_2(edges, emb);
      case 3: if(find_generic_3(edges, emb)) return true; else break;
      case 4: if(find_generic_4(edges, emb)) return true; else break;
      default: break;
    }
    cell_cache<pegasus_spec> cells(topo, nodes, edges);
    bundle_cache<pegasus_spec> bundles(cells);
    size_t minw = (size + 1)/2;
    size_t maxw = min(cells.dim[0], cells.dim[1]);
    if(max_length == 0)
        max_length = maxw+12;
    size_t runs = 0;
    //this loop first looks for any embedding at all (maxw+12 is overkill)
    for(size_t w = minw; w <= maxw; w++) {
        vector<vector<size_t>> tmp;
        //if we've already found an embedding, try to find one with shorter chains
        clique_cache<pegasus_spec> rects(cells, bundles, w, max_length-1);
        runs++;
        if(rects.extract_solution(tmp)) {
            //if we find an embedding, check that it's big enough
            if(tmp.size() >= size) {
                //if it's big enough, find the max (necessary) chainlength
                size_t tlen = get_maxlen(tmp, size);
                if(tlen < max_length) {
                    emb = tmp;
                    max_length = tlen;
                    maxw = min(maxw, w+6);
                    w--; //re-do this width until we stop improving
                }
            }
        } //if no embeddings were found, keep searching
    }
    return emb.size() >= size;
}

}
