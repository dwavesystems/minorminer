// Copyright 2022-2023 D-Wave Systems Inc.
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
//
// Excerpted from and/or inspired by implementations found in the Boost Library
// see external/boost/BOOST_LICENSE or https://www.boost.org/LICENSE_1_0.txt

#pragma once

#include <boost/bimap/unordered_set_of.hpp>
#include <unordered_map>
#include <tuple>

namespace boost {

template<typename L, typename R, typename Alloc>
class bimap;

template<typename L, typename R, typename Alloc>
class bimap<boost::bimaps::unordered_set_of<L>, boost::bimaps::unordered_set_of<R>, Alloc> {
  public:
    std::unordered_map<L, R> left;
    std::unordered_map<R, L> right;
    typedef typename std::pair<L, R> value_type;

    void insert(const value_type v) {
        left.insert(std::make_pair(v.first, v.second));
        right.insert(std::make_pair(v.second, v.first));
    }

    bimap() {}
};

}
