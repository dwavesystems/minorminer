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

#include "util.hpp"
#include "cell_cache.hpp"
#include "bundle_cache.hpp"
#include "clique_cache.hpp"
#include "small_cliques.hpp"
#include "topo_cache.hpp"


namespace busclique {


template<typename T>
size_t get_maxlen(vector<T> &emb, size_t size) {
    auto cmp = [](T a, T b) { return a.size() < b.size(); };
    std::sort(emb.begin(), emb.end(), cmp);
    return emb[size-1].size();
}

template<typename topo_spec>
bool find_clique_nice(const cell_cache<topo_spec> &,
                      size_t size,
                      vector<vector<size_t>> &emb,
                      size_t &min_width,
                      size_t &max_width,
                      size_t &max_length);

template<>
bool find_clique_nice(const cell_cache<chimera_spec> &cells,
                      size_t size,
                      vector<vector<size_t>> &emb,
                      size_t &,
                      size_t &,
                      size_t &max_length) {
    bundle_cache<chimera_spec> bundles(cells);
    size_t shore = cells.topo.shore;
    if(size <= shore)
        for(size_t y = 0; y < cells.topo.dim[0]; y++)
            for(size_t x = 0; x < cells.topo.dim[1]; x++)
                if (bundles.score(y,x,y,y,x,x) >= size) {
                    bundles.inflate(y,x,y,y,x,x,emb);
                    return true;
                }
    size_t minw = (size + shore - 1)/shore;
    size_t maxw = min(cells.topo.dim[0], cells.topo.dim[1]);
    if (max_length > 0) maxw = min(max_length - 1, maxw);
    for(size_t width = minw; width <= maxw; width++) {
        clique_cache<chimera_spec> rects(cells, bundles, width);
        clique_iterator<chimera_spec> iter(cells, rects);
        if(iter.next(emb)) {
            if(emb.size() < size)
                emb.clear();
            else {
                max_length = width + 1;
                return true;
            }
        }
    }
    return false;
}

template<>
bool find_clique_nice(const cell_cache<pegasus_spec> &cells,
                      size_t size,
                      vector<vector<size_t>> &emb,
                      size_t &,
                      size_t &,
                      size_t &max_length) {
    bundle_cache<pegasus_spec> bundles(cells);
    size_t minw = (size + 1)/2;
    size_t maxw = cells.topo.dim[0];
    if(max_length == 0) {
        //naive first-pass: search for the first embedding with any max chainlength
        for(; minw <= maxw; minw++) {
            vector<vector<size_t>> tmp;
            clique_cache<pegasus_spec> rects(cells, bundles, minw);
            if(rects.extract_solution(tmp))
                //if we find an embedding, check that it's big enough
                if(tmp.size() >= size) {
                    emb = tmp;
                    max_length = get_maxlen(tmp, size);
                    maxw = min(maxw, minw + 6);
                    break;
                }
        }
        if(minw > maxw) return false;
    } else {
        maxw = min(cells.topo.dim[0], (max_length)*6);
    }
    //we've already found an embedding; now try to find one with shorter chains
    for(size_t w = minw; w <= maxw; w++) {
        auto check_length = [&bundles, max_length](size_t yc, size_t xc,
                                                   size_t y0, size_t y1,
                                                   size_t x0, size_t x1){
            return bundles.length(yc,xc,y0,y1,x0,x1) < max_length; 
        };
        clique_cache<pegasus_spec> rects(cells, bundles, w, check_length);
        vector<vector<size_t>> tmp;
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

template<typename topo_spec>
bool find_clique(const topo_spec &topo,
                 const vector<size_t> &nodes,
                 const vector<pair<size_t, size_t>> &edges,
                 size_t size,
                 vector<vector<size_t>> &emb) {
    constexpr bool pegasus = std::is_same<topo_spec, pegasus_spec>::value;
    switch(size) {
      case 0: return true;
      case 1: return find_generic_1(nodes, emb);
      case 2: return find_generic_2(edges, emb);
      case 3: if(pegasus && find_generic_3(edges, emb)) return true; else break;
      case 4: if(pegasus && find_generic_4(edges, emb)) return true; else break;
      default: break;
    }
    topo_cache<topo_spec> topology(topo, nodes, edges);
    return find_clique(topology, size, emb);
}

template<typename topo_spec>
bool find_clique(topo_cache<topo_spec> &topology,
                 size_t size,
                 vector<vector<size_t>> &emb) {
    topology.reset();
    size_t max_length = 0;
    size_t min_width = 0;
    size_t max_width = 0;
    vector<vector<size_t>> _emb;
    if (find_clique_nice(topology.cells, size, _emb, max_length, min_width, max_width))
        emb = _emb;
    while(topology.next()) {
        if (find_clique_nice(topology.cells, size, _emb, max_length, min_width, max_width))
            emb = _emb;
    }
    return emb.size() >= size;
}

template<typename topo_spec>
bool find_clique_nice(const topo_spec &topo,
                      const vector<size_t> &nodes,
                      const vector<pair<size_t, size_t>> &edges,
                      size_t size,
                      vector<vector<size_t>> &emb) {
    const cell_cache<topo_spec> cells(topo, nodes, edges);
    size_t _0, _1, _2;
    return find_clique_nice(cells, size, emb, _0, _1, _2);
}

template<typename topo_spec>
void short_clique(const topo_spec &,
                  const vector<size_t> &nodes,
                  const vector<pair<size_t, size_t>> &edges,
                  vector<vector<size_t>> &emb) {
    constexpr bool pegasus = std::is_same<topo_spec, pegasus_spec>::value;
    if(pegasus && find_generic_4(edges, emb))       return;
    else if(pegasus && find_generic_3(edges, emb))  return;
    else if(find_generic_2(edges, emb))             return;
    else if(find_generic_1(nodes, emb))             return;
}

template<typename topo_spec>
void best_cliques(topo_cache<topo_spec> &topology,
                  vector<vector<vector<size_t>>> &embs,
                  vector<vector<size_t>> &emb_1) {
    embs.clear();
    embs.push_back(vector<vector<size_t>>{});
    embs.push_back(emb_1);
    topology.reset();
    do {
        clique_yield_cache<topo_spec> cliques(topology.cells);
        size_t chainlength = 0;
        for(auto &_emb: cliques.embeddings()) {
            while(embs.size() <= chainlength)
                embs.emplace_back(0);
            if(_emb.size() > embs[chainlength].size())
                embs[chainlength] = _emb;
            chainlength++;
        }
    } while(topology.next());
}

}
