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
#include "cell_cache.hpp"
#include <random>
#include <math.h>

namespace busclique {

template<typename topo_spec>
class topo_cache {
  public:
    const topo_spec topo;
  private:
    fat_pointer<uint8_t> nodemask;
    fat_pointer<uint8_t> edgemask;
    fat_pointer<uint8_t> badmask;
    vector<pair<size_t, size_t>> bad_edges;
    uint64_t mask_num;
    uint64_t mask_bound;
    double log_mask_bound;

    fastrng rng;
    //this is a little hackish way to keep everything const & construct cells in-place
    class _initializer_tag {};
    _initializer_tag _init;

    uint8_t *child_nodemask;
    uint8_t *child_edgemask;
  public:
    const cell_cache<topo_spec> cells;

  public:
    //prevent double-frees by forbidding moving & copying
    topo_cache(const topo_cache &) = delete;
    topo_cache(topo_cache &&) = delete;
    ~topo_cache() {
        if (nodemask != child_nodemask) {
            delete []child_nodemask;
            child_nodemask = nullptr;
        }
        if (edgemask != child_edgemask) {
            delete []child_edgemask;
            child_edgemask = nullptr;
        }
    }

    topo_cache(const topo_spec t, const vector<size_t> &nodes,
               const vector<pair<size_t, size_t>> &edges) :
               topo(t),
               nodemask(t.num_cells(), 0),
               edgemask(t.num_cells(), 0),
               badmask(t.num_cells()*t.shore, 0),
               bad_edges(), mask_num(0), mask_bound(64), log_mask_bound(6),
               rng(topo.seed), _init(_initialize(nodes, edges)),
               cells(t, child_nodemask, child_edgemask) {}

    void reset() {
        if(mask_num > 0) {
            mask_num = 0;
            rng = fastrng(topo.seed);
            next();
        }
    }

    template<typename serialize_tag>
    size_t serialize(serialize_tag, uint8_t *output) const {
        return topo.serialize(serialize_tag{}, output, nodemask, edgemask, badmask);
    }

    vector<size_t> fragment_nodes(const uint8_t *nmask = nullptr) const {
        if (nmask == nullptr) nmask = nodemask;
        vector<size_t> nodes;
        size_t q = 0;
        for (size_y y = 0; y < topo.dim_y; y++) {
            for (size_x x = 0; x < topo.dim_x; x++) {
                for (uint8_t k = 0; k < topo.shore; k++) {
                    if (nmask[topo.cell_index(0, vert(x), vert(y))]&mask_bit[k])
                        nodes.push_back(q);
                    minorminer_assert(q == topo.chimera_linear(y, x, 0, k));
                    q++;
                }
                for (uint8_t k = 0; k < topo.shore; k++) {
                    if (nmask[topo.cell_index(1, horz(y), horz(x))]&mask_bit[k])
                        nodes.push_back(q);
                    minorminer_assert(q == topo.chimera_linear(y, x, 1, k));
                    q++;
                }
            }
        }
        return nodes;
    }
    
    vector<pair<size_t, size_t>> fragment_edges(const uint8_t *nmask = nullptr, const uint8_t *emask = nullptr) const {
        if (nmask == nullptr) nmask = nodemask;
        if (emask == nullptr) emask = edgemask;
        vector<pair<size_t, size_t>> edges;
        size_t q = 0;
        for (size_y y = 0; y < topo.dim_y; y++) {
            for (size_x x = 0; x < topo.dim_x; x++) {
                for (uint8_t k = 0; k < topo.shore; k++) {
                    if (emask[topo.cell_index(0, vert(x), vert(y))]&mask_bit[k])
                        edges.emplace_back(q, topo.chimera_linear(y-1_y, x, 0, k));
                    if (nmask[topo.cell_index(0, vert(x), vert(y))]&mask_bit[k]) {
                        for (uint8_t k1 = 0; k1 < topo.shore; k1++) {
                            if (nmask[topo.cell_index(1, horz(y), horz(x))]&mask_bit[k1]&~badmask[q])
                                edges.emplace_back(q, topo.chimera_linear(y, x, 1, k1));
                        }
                    }
                    minorminer_assert(q == topo.chimera_linear(y, x, 0, k));
                    q++;
                }
                for (uint8_t k = 0; k < topo.shore; k++) {
                    if (emask[topo.cell_index(1, horz(y), horz(x))]&mask_bit[k])
                        edges.emplace_back(q, topo.chimera_linear(y, x-1_x, 1, k));
                    minorminer_assert(q == topo.chimera_linear(y, x, 1, k));
                    q++;
                }
            }
        }
        return edges;
    }

    void set_mask_bound(uint64_t bound) {
        mask_bound = bound;
        log_mask_bound = log2(static_cast<long double>(mask_bound));
    }

  private:
    //this is a funny hack used to construct cells in-place:
    // _initializer_tag is an empty (size zero) struct, so this is "zero-cost"
    _initializer_tag _initialize(const vector<size_t> &nodes,
                                 const vector<pair<size_t, size_t>> &edges) {
        if(std::is_same<topo_spec, chimera_spec>::value) {
            topo.process_nodes(nodemask, nullptr, badmask, nodes);
            topo.process_edges(edgemask, badmask, edges);
        } else {
            topo.process_nodes(nodemask, edgemask, badmask, nodes);
            topo.process_edges(edgemask, badmask, edges);
        }
        topo.finish_badmask(nodemask, badmask);
        compute_bad_edges();
        if(bad_edges.size() > 0) {
            child_nodemask = new uint8_t[topo.num_cells()];
            child_edgemask = new uint8_t[topo.num_cells()];
        } else {
            child_nodemask = nodemask;
            child_edgemask = edgemask;
        }
        next();
        return _initializer_tag {};
    }

    void compute_bad_edges() {
        size_t q = 0;
        for(size_y y = 0; y < topo.dim_y; y++)
            for(size_x x = 0; x < topo.dim_x; x++) {
                //only iterate over the vertical qubits to avoid duplication
                for(uint8_t k = 0; k < topo.shore; k++) {
                    uint8_t mask = badmask[q];
                    while(mask) {
                        uint8_t badk = first_bit[mask];
                        mask ^= mask_bit[badk];
                        bad_edges.emplace_back(q, topo.chimera_linear(y, x, 1, badk));
                    }
                    minorminer_assert(q == topo.chimera_linear(y, x, 0, k));
                    q++;
                }
                //skip over those horizontal qubits
                q += topo.shore;
            }
    }

  public:
    bool next() {
        if(bad_edges.size() == 0) {
            if (mask_num == 0) {
                mask_num++;
                return true;
            } else {
                return false;
            }
        }
        vector<size_t> bad_nodes;
        if (static_cast<double>(bad_edges.size()) <= log_mask_bound) {
            if (mask_num < (1_u64 << bad_edges.size())) {
                for(size_t i = bad_edges.size(); i--;) {
                    if(mask_num & (1_u64 << i))
                        bad_nodes.push_back(bad_edges[i].first);
                    else
                        bad_nodes.push_back(bad_edges[i].second);
                }
                mask_num++;
            } else {
                return false;
            }
        } else {
            if(mask_num < mask_bound) mask_num++;
            else return false;
            //this is a somewhat ad-hoc, unoptimized implementation.
            std::shuffle(bad_edges.begin(), bad_edges.end(), rng);
            std::map<size_t, std::set<size_t>> adj;
            for(auto &e : bad_edges) {
                auto a = adj.find(e.first);
                auto b = adj.find(e.second);
                if(a == adj.end()) a = adj.emplace(e.first, _emptyset).first;
                if(b == adj.end()) b = adj.emplace(e.second, _emptyset).first;
                std::set<size_t> &A = (*a).second;
                std::set<size_t> &B = (*b).second;
                B.emplace(e.first);
                A.emplace(e.second);
            }
            auto degree = [&adj](size_t x) { return adj[x].size(); };
            std::sort(bad_edges.begin(), bad_edges.end(), 
                     [&degree](pair<size_t, size_t> e, pair<size_t, size_t> f) {
                         size_t ekey = degree(e.first) + degree(e.second);
                         size_t fkey = degree(f.first) + degree(f.second);
                         return ekey < fkey;
                     });
            std::set<size_t> cover;
            for(auto &e: bad_edges) {
                size_t x;
                if (cover.count(e.first) + cover.count(e.second)) continue;
                else if (degree(e.first) < degree(e.second)) x = e.second;
                else if (degree(e.first) > degree(e.second)) x = e.first;
                else if (rng()&1) { x = e.second; }
                else              { x = e.first;  }
                cover.insert(x);
                for(auto &y: adj[x])
                    adj[y].erase(x);
                adj[x].clear();
            }
            for(auto &q: cover) bad_nodes.push_back(q);
        }
        memcpy(child_nodemask, nodemask, topo.num_cells());
        memcpy(child_edgemask, edgemask, topo.num_cells());

        for(auto &q: bad_nodes) {
            size_y y;
            size_x x;
            bool u;
            uint8_t k;
            topo.linear_chimera(q, y, x, u, k);
            child_nodemask[topo.cell_index(y, x, u)] &= ~mask_bit[k];
            child_edgemask[topo.cell_index(y, x, u)] &= ~mask_bit[k];
            if(u) {
                if (x+1_x < topo.dim_x) 
                    child_edgemask[topo.cell_index(y, x+1_x, u)] &= ~mask_bit[k];
            } else {
                if (y+1_y < topo.dim_y) 
                    child_edgemask[topo.cell_index(y+1_y, x, u)] &= ~mask_bit[k];
            }
        }
        return true;
    }
};

}
