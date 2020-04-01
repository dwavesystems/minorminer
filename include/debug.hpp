#pragma once
#include "util.hpp"
namespace find_embedding {

struct disable_asserts {
    static inline void assertion(bool /*statement*/) {}
};

struct enable_asserts {
    static inline void assertion(bool statement) {
        if (!statement) throw MinorMinerException();
    }
};

}  // namespace find_embedding