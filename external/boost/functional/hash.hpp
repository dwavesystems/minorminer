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

namespace boost {

template <typename T>
inline void hash_combine(std::size_t & seed, const T value) {
  std::hash<T> h;
  hash_combine(seed, h(value));
}

template <>
inline void hash_combine(std::size_t & seed, const std::size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}
