#pragma once
#include<stdexcept>
#include<string>

namespace minorminer {

class MinorMinerException : public std::runtime_error {
  public:
    MinorMinerException(const std::string& m = "find embedding exception") : std::runtime_error(m) {}
};

}
