#include "util.hpp"
#include "cell_cache.hpp"
#include "bundle_cache.hpp"
#include "clique_cache.hpp"

namespace busclique {

template<typename topo_spec>
bool find_clique(topo_spec &topo, 
                 const vector<size_t> &nodes,
                 const vector<pair<size_t, size_t>> &edges,
                 size_t size,
                 vector<vector<size_t>> &emb) {
    cell_cache<topo_spec> chim(topo, nodes, edges);
    bundle_cache<topo_spec> bundles(chim);

    size_t shore = chim.shore;
    if(size <= shore) {
        size_t by, bx, score=0;
        for(size_t y = 0; y < chim.dim[0]; y++)
            for(size_t x = 0; x < chim.dim[1]; x++) {
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
    for(size_t width = (size + shore - 1)/shore; 
            width <= min(chim.dim[0], chim.dim[1]); width++) {
        clique_cache<topo_spec> rects(chim, bundles, width, 9999);
        if(clique_iterator<topo_spec>(chim, rects).next(emb)) {
            if(emb.size() < size) emb.clear();
            else return true;
        }
    }
    return false;
}

}
