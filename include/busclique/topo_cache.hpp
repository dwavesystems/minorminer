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

namespace busclique {

template<typename topo_spec>
class topo_cache {
  public:
    const topo_spec topo;
  private:
    uint8_t *nodemask;
    uint8_t *edgemask;
    uint8_t *badmask;
    vector<pair<size_t, size_t>> bad_edges;
    uint8_t mask_num;

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
        if(nodemask != nullptr) { 
            if (nodemask == child_nodemask) child_nodemask = nullptr; 
            delete []nodemask; nodemask = nullptr; 
        }
        if(edgemask != nullptr) {
            if (edgemask == child_edgemask) child_edgemask = nullptr;
            delete []edgemask; edgemask = nullptr; 
        }
        if(badmask != nullptr) { delete []badmask; badmask = nullptr; }
        if(child_nodemask != nullptr) { delete []child_nodemask; child_nodemask = nullptr; }
        if(child_edgemask != nullptr) { delete []child_edgemask; child_edgemask = nullptr; }
    }

    topo_cache(const pegasus_spec t, const vector<size_t> &nodes,
               const vector<pair<size_t, size_t>> &edges) :
               topo(t),
               nodemask(new uint8_t[t.num_cells()]{}),
               edgemask(new uint8_t[t.num_cells()]{}),
               badmask(new uint8_t[t.num_cells()*t.shore]{}),
               bad_edges(), mask_num(0), _init(_initialize(nodes, edges)),
               cells(t, child_nodemask, child_edgemask) {}

    topo_cache(const chimera_spec t, const vector<size_t> &nodes,
               const vector<pair<size_t, size_t>> &edges) :
               topo(t),
               nodemask(new uint8_t[t.num_cells()]{}),
               edgemask(new uint8_t[t.num_cells()]{}),
               badmask(new uint8_t[t.num_cells()*t.shore]{}),
               bad_edges(), mask_num(0), _init(_initialize(nodes, edges)),
               cells(t, child_nodemask, child_edgemask) {}

    void reset() {
        if(mask_num > 0) {
            mask_num = 0;
            next();
        }
    }

  private:
    //this is a funny hack used to construct cells in-place:
    // _initializer_tag is an empty (size zero) struct, so this is "zero-cost"
    _initializer_tag _initialize(const vector<size_t> &nodes,
                                 const vector<pair<size_t, size_t>> &edges) {
        if(std::is_same<topo_spec, pegasus_spec>::value) {
            topo.process_nodes(nodemask, edgemask, badmask, nodes);
            topo.process_edges(edgemask, badmask, edges);
        } else {
            topo.process_nodes(nodemask, nullptr, badmask, nodes);
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
        for(size_t y = 0, rowbase = 0; y < topo.dim[0]; y++, rowbase += topo.shore)
            for(size_t x = 0; x < topo.dim[1]; x++) {
                size_t qbase = q;
                //only iterate over the vertical qubits to avoid duplication
                for(size_t k = 0, row = rowbase; k < topo.shore; k++, q++, row++) {
                    uint8_t mask = badmask[q];
                    while(mask) {
                        uint8_t badk = first_bit[mask];
                        mask ^= mask_bit[badk];
                        bad_edges.emplace_back(q, qbase + topo.shore + badk);
                    }
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
        if (bad_edges.size() < 7) {
            if (mask_num < mask_subsets[bad_edges.size()]) {
                for(size_t i = bad_edges.size(); i--;) {
                    if(mask_num & mask_bit[i])
                        bad_nodes.push_back(bad_edges[i].first);
                    else
                        bad_nodes.push_back(bad_edges[i].second);
                }
                mask_num++;
            } else {
                return false;
            }
        } else {
            //maaaaybe we want to hand control of this parameter to the user?
            if(mask_num < 64) mask_num++;
            else return false;
            //this is a somewhat ad-hoc, unoptimized implementation.  we should
            //take a proper seed from the user, etc.
            std::random_device r;
            std::ranlux48 rng(r());
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
            size_t u, w, z, k;
            topo.linemajor(q, u, w, z, k);
            child_nodemask[topo.cell_addr(u, w, z)] &= ~mask_bit[k];
            child_edgemask[topo.cell_addr(u, w, z)] &= ~mask_bit[k];
            if(z + 1 < topo.dim[u])
                child_edgemask[topo.cell_addr(u, w, z + 1)] &= ~mask_bit[k];
        }
        return true;
    }
};

}
