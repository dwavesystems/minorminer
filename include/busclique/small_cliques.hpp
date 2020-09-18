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

namespace busclique {

//find a 1-clique... uh... node... in a generic graph
bool find_generic_1(const vector<size_t> &nodes,
                    vector<vector<size_t>> &emb) {
    if(nodes.size() == 0)
        return false;
    emb.emplace_back(1, nodes[0]);
    return true;
}

//find a 2-clique... uh... edge... in a generic graph
bool find_generic_2(const vector<pair<size_t, size_t>> &edges,
                    vector<vector<size_t>> &emb) {
    if(edges.size() == 0)
        return false;
    emb.emplace_back(1, edges[0].first);
    emb.emplace_back(1, edges[0].second);
    return true;
}

//find a 3-clique in a generic graph
bool find_generic_3(const vector<pair<size_t, size_t>> &edges,
                    vector<vector<size_t>> &emb) {
    //this is very much not optimized, 'cause it's a silly case
    std::map<size_t, std::set<size_t>> adj;
    for(auto &e : edges) {
        auto a = adj.find(e.first);
        auto b = adj.find(e.second);
        if(a == adj.end()) a = adj.emplace(e.first, _emptyset).first;
        if(b == adj.end()) b = adj.emplace(e.second, _emptyset).first;
        std::set<size_t> &A = (*a).second;
        std::set<size_t> &B = (*b).second;
        for(auto &p : A) {
            if(B.count(p)) {
                emb.emplace_back(1, e.first);
                emb.emplace_back(1, e.second);
                emb.emplace_back(1, p);
                return true;
            }
        }
        B.emplace(e.first);
        A.emplace(e.second);
    }
    return false;
}

//find a 4-clique in a generic graph
bool find_generic_4(const vector<pair<size_t, size_t>> &edges,
                    vector<vector<size_t>> &emb) {
    //this is very much not optimized, 'cause it's a silly case
    std::map<size_t, std::set<size_t>> adj;
    for(auto &e : edges) {
        auto a = adj.find(e.first);
        auto b = adj.find(e.second);
        if(a == adj.end()) a = adj.emplace(e.first, _emptyset).first;
        if(b == adj.end()) b = adj.emplace(e.second, _emptyset).first;
        std::set<size_t> &A = (*a).second;
        std::set<size_t> &B = (*b).second;

        for(auto &p : A) {
            if(B.count(p) == 0) continue;
            for(auto &q : A) {
                if(p < q || B.count(q) == 0) continue;
                // here we've found the edges a~b, a~p, a~q, b~p, b~q
                // so to finish the clique we only need to know p~q
                auto z = adj.find(q);
                //q should be in there, but just in case?
                if(z != adj.end() && (*z).second.count(p)) {
                    emb.emplace_back(1, e.first);
                    emb.emplace_back(1, e.second);
                    emb.emplace_back(1, p);
                    emb.emplace_back(1, q);
                    return true;
                }
            }
        }
        B.emplace(e.first);
        A.emplace(e.second);
    }
    return false;
}


}
