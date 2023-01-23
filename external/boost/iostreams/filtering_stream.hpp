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

#include <iostream>
#include <streambuf>
namespace boost::iostreams {

class filtering_ostream: public std::basic_ostream<char> {
    class fake_streambuf: public std::basic_streambuf<char> {
      public:
        fake_streambuf() : std::basic_streambuf<char>() {}
    };
    fake_streambuf sb;
  public:
    filtering_ostream() : sb(), basic_ostream(&sb) {}
    template<typename T>
    void push(T) { throw std::runtime_error("filtering_ostream is not implemented"); }
};

}
