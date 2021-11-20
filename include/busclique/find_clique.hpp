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

template<typename clique_cache_t>
size_t get_sol(const clique_cache_t &rects, vector<vector<size_t>> &emb, size_t size) {
    if(rects.extract_solution(emb)) {
        if (emb.size() >= size) {
            return get_maxlen(emb, size);
        }
    }
    return numeric_limits<size_t>::max();
}


template<typename cells_t>
bool find_clique_short(const cells_t &cells, const size_t size, vector<vector<size_t>> &emb, size_t &max_length) {
    for(size_y y = 0; y < cells.topo.dim_y; y++)
        for(size_x x = 0; x < cells.topo.dim_x; x++)
            if (cells.score(y,x) >= size) {
                emb.clear();
                max_length = 1;
                cells.inflate(y, x, emb);
                return true;
            }
    return false;
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
    size_t shore = cells.topo.shore;
    if(size <= shore && find_clique_short(cells, size, emb, max_length))
        return true;
    else if (max_length == 1)
        return false;
    bundle_cache<chimera_spec> bundles(cells);
    size_t minw = (size + shore - 1)/shore;
    size_t maxw = coordinate_converter::min(cells.topo.dim_y, cells.topo.dim_x);
    maxw = min(max_length - 1, maxw);
    for(size_t width = minw; width <= maxw; width++) {
        clique_cache<chimera_spec> rects(cells, bundles, width);
        vector<vector<size_t>> temp_emb;
        size_t l = get_sol(rects, temp_emb, size);
        if (l < max_length) {
            emb = temp_emb;
            max_length = l;
            return true;
        }
    }
    return false;
}

template<>
bool find_clique_nice(const cell_cache<zephyr_spec> &cells,
                      size_t size,
                      vector<vector<size_t>> &emb,
                      size_t &,
                      size_t &,
                      size_t &max_length) {
    size_t shore = cells.topo.shore;
    if(size <= shore && find_clique_short(cells, size, emb, max_length))
        return true;
    else if (max_length == 1)
        return false;
    bundle_cache<zephyr_spec> bundles(cells);
    size_t minw = (size + shore - 1)/shore;
    size_t maxw = coordinate_converter::min(cells.topo.dim_y, cells.topo.dim_x);
    if (max_length < numeric_limits<size_t>::max()) maxw = min(2*max_length+1, maxw);
    for(size_t width = minw; width <= maxw; width++) {
        clique_cache<zephyr_spec> rects(cells, bundles, width);
        vector<vector<size_t>> temp_emb;
        size_t l = get_sol(rects, temp_emb, size);
        if (l < max_length) {
            max_length = l;
            emb = temp_emb;
            return true;
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
    size_t maxw = coordinate_converter::min(cells.topo.dim_y, cells.topo.dim_x);

    for(size_t w = minw; w <= maxw; w++) {
        size_t w_lengthmin, w_lengthmax;
        clique_yield_cache<pegasus_spec>::get_length_range(cells.topo, w, w_lengthmin, w_lengthmax);
        if (w_lengthmin < max_length) {
            //check for any embedding at this length
            vector<vector<size_t>> tmp;
            clique_cache<pegasus_spec> rects(cells, bundles, w);
            if(rects.extract_solution(tmp) && tmp.size() >= size) {
                size_t length = get_maxlen(tmp, size);
                if (length < max_length)
                    emb = tmp;
                max_length = min(length, max_length);
                maxw = min(maxw, w+6);
            } else {
                //no embeddings here; skip to the next minw;
                continue;
            }
        } else {
            break;
        }
        w_lengthmax = max(max_length, w_lengthmax);
        for (size_t length = w_lengthmin; length < w_lengthmax; length++) {
            auto check_length = [&bundles, length](size_y yc, size_x xc,
                                                   size_y y0, size_y y1,
                                                   size_x x0, size_x x1){
                return bundles.length(yc,xc,y0,y1,x0,x1) <= length;
            };
            clique_cache<pegasus_spec> rects(cells, bundles, w, check_length);
            vector<vector<size_t>> tmp;
            if(rects.extract_solution(tmp)) {
                if(tmp.size() >= size) {
                    //if it's big enough, find the max (necessary) chainlength
                    size_t tlen = get_maxlen(tmp, size);
                    if(tlen < max_length)
                        emb = tmp;
                    max_length = min(length, max_length);
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
    if (size <= topo_spec::clique_number) {
        switch(size) {
          case 0: return true;
          case 1: return find_generic_1(nodes, emb);
          case 2: return find_generic_2(edges, emb);
          case 3: if(find_generic_3(edges, emb)) return true; else break;
          case 4: if(find_generic_4(edges, emb)) return true; else break;
          default: break;
        }
    }
    topo_cache<topo_spec> topology(topo, nodes, edges);
    return find_clique(topology, size, emb);
}

template<typename topo_spec>
bool find_clique(topo_cache<topo_spec> &topology,
                 size_t size,
                 vector<vector<size_t>> &emb) {
    topology.reset();
    size_t max_length = numeric_limits<size_t>::max();
    size_t min_width = 0;
    size_t max_width = 0;
    vector<vector<size_t>> _emb;
    if (find_clique_nice(topology.cells, size, _emb, min_width, max_width, max_length))
        emb = _emb;
    while(topology.next()) {
        if (find_clique_nice(topology.cells, size, _emb, min_width, max_width, max_length))
            emb = _emb;
    }
    return emb.size() >= size;
}

template<typename topo_spec>
void short_clique(const topo_spec &,
                  const vector<size_t> &nodes,
                  const vector<pair<size_t, size_t>> &edges,
                  vector<vector<size_t>> &emb) {
    constexpr size_t n = topo_spec::clique_number;
    static_assert(n > 1, "topologies with edges have clique number at least 2");
    if(n >= 4 && find_generic_4(edges, emb))       return;
    else if( n >= 3 && find_generic_3(edges, emb)) return;
    else if(find_generic_2(edges, emb))            return;
    else if(find_generic_1(nodes, emb))            return;
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
