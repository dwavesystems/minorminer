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

#pragma once
#include <algorithm>
#include <chrono>
#include <iterator>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include "../debug.hpp"
#include "../errors.hpp"
#include "../fastrng.hpp"
#include "pairing_queue.hpp"

namespace find_embedding {
// Import some things from the std library
using std::default_random_engine;
using std::map;
using std::max;
using std::min;
using std::mutex;
using std::numeric_limits;
using std::pair;
using std::shared_ptr;
using std::string;
using std::thread;
using std::uniform_int_distribution;
using std::unordered_map;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using fastrng::fastrng;


// Select some default structures and types
using distance_t = long long int;
constexpr distance_t max_distance = numeric_limits<distance_t>::max();
using RANDOM = fastrng;
using clock = std::chrono::high_resolution_clock;
template <typename P>
using min_queue = std::priority_queue<priority_node<P, min_heap_tag>>;
template <typename P>
using max_queue = std::priority_queue<priority_node<P, max_heap_tag>>;

using distance_queue = pairing_queue<priority_node<distance_t, min_heap_tag>>;

using minorminer::MinorMinerException;

class ProblemCancelledException : public MinorMinerException {
  public:
    ProblemCancelledException(const string& m = "embedding cancelled by keyboard interrupt") : MinorMinerException(m) {}
};

class TimeoutException : public MinorMinerException {
  public:
    TimeoutException(const string& m = "embedding timed out") : MinorMinerException(m) {}
};

class CorruptParametersException : public MinorMinerException {
  public:
    CorruptParametersException(const string& m = "chain inputs are corrupted") : MinorMinerException(m) {}
};

class BadInitializationException : public MinorMinerException {
  public:
    BadInitializationException(const string& m = "bad embedding used with skip_initialization")
            : MinorMinerException(m) {}
};

class CorruptEmbeddingException : public MinorMinerException {
  public:
    CorruptEmbeddingException(const string& m = "chains may be invalid") : MinorMinerException(m) {}
};

//! Interface for communication between the library and various bindings.
//!
//! Any bindings of this library need to provide a concrete subclass.
class LocalInteraction {
  public:
    virtual ~LocalInteraction() {}
    //! Print a message through the local output method
    void displayOutput(int loglevel, const string& msg) const { displayOutputImpl(loglevel, msg); }

    //! Print an error through the local output method
    void displayError(int loglevel, const string& msg) const { displayErrorImpl(loglevel, msg); }

    //! Check if someone is trying to cancel the embedding process
    int cancelled(const clock::time_point stoptime) const {
        if (cancelledImpl()) throw ProblemCancelledException();
        if (timedOutImpl(stoptime)) throw TimeoutException();
        return 0;
    }

  private:
    //! Print the string to a binding specified sink
    virtual void displayOutputImpl(int loglevel, const string&) const = 0;

    //! Print the error to a binding specified sink
    virtual void displayErrorImpl(int loglevel, const string&) const = 0;

    //! Check if the embedding process has timed out.
    virtual bool timedOutImpl(const clock::time_point stoptime) const { return clock::now() >= stoptime; }

    //! Check if someone has tried to cancel the embedding process
    virtual bool cancelledImpl() const = 0;
};

typedef shared_ptr<LocalInteraction> LocalInteractionPtr;

//! Set of parameters used to control the embedding process.
class optional_parameters {
  public:
    //! actually not controlled by user, not initialized here, but initialized in Python, MATLAB, C wrappers level
    LocalInteractionPtr localInteractionPtr;
    int max_no_improvement = 10;
    RANDOM rng;
    //! Number of seconds before the process unconditionally stops
    double timeout = 1000;
    double max_beta = numeric_limits<double>::max();
    int tries = 10;
    int verbose = 0;
    bool interactive = false;
    int inner_rounds = numeric_limits<int>::max();
    int max_fill = numeric_limits<int>::max();
    bool return_overlap = false;
    int chainlength_patience = 2;
    int threads = 1;
    bool skip_initialization = false;
    map<int, vector<int>> fixed_chains;
    map<int, vector<int>> initial_chains;
    map<int, vector<int>> restrict_chains;

    //! duplicate all parameters but chain hints,
    //! and seed a new rng.  this vaguely peculiar behavior is
    //! utilized to spawn parameters for component subproblems
    optional_parameters(optional_parameters& p, map<int, vector<int>> fixed_chains,
                        map<int, vector<int>> initial_chains, map<int, vector<int>> restrict_chains)
            : localInteractionPtr(p.localInteractionPtr),
              max_no_improvement(p.max_no_improvement),
              rng(p.rng()),
              timeout(p.timeout),
              max_beta(p.max_beta),
              tries(p.tries),
              verbose(p.verbose),
              interactive(p.interactive),
              inner_rounds(p.inner_rounds),
              max_fill(p.max_fill),
              return_overlap(p.return_overlap),
              chainlength_patience(p.chainlength_patience),
              threads(p.threads),
              skip_initialization(p.skip_initialization),
              fixed_chains(fixed_chains),
              initial_chains(initial_chains),
              restrict_chains(restrict_chains) {
#ifndef CPPDEBUG
        if (verbose >= 4)
            throw CorruptParametersException(
                    "this build of minorminer only supports verbose=0, 1, 2 or 3.  "
                    "build with CPPDEBUG=1 for debugging output");
#endif
    }
    //^^leave this constructor by the declarations

  public:
    template <typename... Args>
    void print_out(int loglevel, const char* format, Args... args) const {
        char buffer[1024];
        snprintf(buffer, 1024, format, args...);
        localInteractionPtr->displayOutput(loglevel, buffer);
    }

    void print_out(int loglevel, const char* format) const {
        localInteractionPtr->displayOutput(loglevel, format);
    }


    template <typename... Args>
    void print_err(int loglevel, const char* format, Args... args) const {
        char buffer[1024];
        snprintf(buffer, 1024, format, args...);
        localInteractionPtr->displayError(loglevel, buffer);\
    }

    void print_err(int loglevel, const char* format) const {
        localInteractionPtr->displayError(loglevel, format);
    }

    template <typename... Args>
    void error(const char* format, Args... args) const {
        print_err(0, format, args...);
    }

    template <typename... Args>
    void major_info(const char* format, Args... args) const {
        print_out(1, format, args...);
    }

    template <typename... Args>
    void minor_info(const char* format, Args... args) const {
        print_out(2, format, args...);
    }

    template <typename... Args>
    void extra_info(const char* format, Args... args) const {
        print_out(3, format, args...);
    }

#ifdef CPPDEBUG
    template <typename... Args>
    void debug(const char* format, Args... args) const {
        print_out(4, format, args...);
    }
#else
    template <typename... Args>
    void debug(const char* /*format*/, Args... /*args*/) const {}
#endif

    optional_parameters() : localInteractionPtr(), rng() {}
    void seed(uint64_t randomSeed) { rng.seed(randomSeed); }
};

//! Fill output with the index of all of the minimum and equal values in input
template <typename T>
void collectMinima(const vector<T>& input, vector<int>& output) {
    output.clear();
    auto lowest_value = input[0];
    int index = 0;
    for (auto& y : input) {
        if (y == lowest_value) {
            output.push_back(index);
        } else if (y < lowest_value) {
            output.clear();
            output.push_back(index);
            lowest_value = y;
        }
        index++;
    }
}
}  // namespace find_embedding
