// Copyright 2017 - 2020 D-Wave Systems Inc.
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

#include <iostream>
#include "../include/find_embedding/find_embedding.hpp"

class MyCppInteractions : public find_embedding::LocalInteraction {
  public:
    bool _canceled = false;
    void cancel() { _canceled = true; }

  private:
    void displayOutputImpl(const std::string& mess) const override { std::cout << mess << std::endl; }
    void displayErrorImpl(const std::string& mess) const override { std::cout << mess << std::endl; }
    virtual bool cancelledImpl() const { return _canceled; }
};

int main() {
    graph::input_graph triangle(3, {0, 1, 2}, {1, 2, 0});
    graph::input_graph square(4, {0, 1, 2, 3}, {1, 2, 3, 0});
    find_embedding::optional_parameters params;
    params.localInteractionPtr.reset(new MyCppInteractions());

    std::vector<std::vector<int>> chains;

    if (find_embedding::findEmbedding(triangle, square, params, chains)) {
        for (auto chain : chains) {
            for (auto var : chain) std::cout << var << " ";
            std::cout << std::endl;
        }
    } else {
        std::cout << "Couldn't find embedding." << std::endl;
    }

    return 0;
}