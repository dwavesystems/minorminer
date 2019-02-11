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
#include "debug.hpp"
#include "fastrng.hpp"
#include "pairing_queue.hpp"

namespace find_embedding {
// Import some things from the std library
using std::default_random_engine;
using std::vector;
using std::string;
using std::shared_ptr;
using std::map;
using std::unordered_map;
using std::pair;
using std::numeric_limits;
using std::uniform_int_distribution;
using std::min;
using std::max;
using std::thread;
using std::mutex;
using std::chrono::duration;
using std::chrono::duration_cast;

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

//! Interface for communication between the library and various bindings.
//!
//! Any bindings of this library need to provide a concrete subclass.
class LocalInteraction {
  public:
    virtual ~LocalInteraction() {}
    //! Print a message through the local output method
    void displayOutput(const string& msg) const { displayOutputImpl(msg); }

    //! Check if someone is trying to cancel the embedding process
    bool cancelled(const clock::time_point stoptime) const {
        if (cancelledImpl()) {
            displayOutput("caught interrupt; embedding cancelled\n");
            return true;
        }
        if (timedOutImpl(stoptime)) {
            displayOutput("embedding timed out\n");
            return true;
        }
        return false;
    }

  private:
    //! Print the string to a binding specified sink
    virtual void displayOutputImpl(const string&) const = 0;

    //! Check if the embedding process has timed out.
    virtual bool timedOutImpl(const clock::time_point stoptime) const { return clock::now() >= stoptime; }

    //! Check if someone has tried to cancel the embedding process
    virtual bool cancelledImpl() const = 0;
};

typedef shared_ptr<LocalInteraction> LocalInteractionPtr;

class MinorMinerException : public std::runtime_error {
  public:
    MinorMinerException(const string& m = "find embedding exception") : std::runtime_error(m) {}
};

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
              inner_rounds(p.inner_rounds),
              max_fill(p.max_fill),
              return_overlap(p.return_overlap),
              chainlength_patience(p.chainlength_patience),
              threads(p.threads),
              skip_initialization(p.skip_initialization),
              fixed_chains(fixed_chains),
              initial_chains(initial_chains),
              restrict_chains(restrict_chains) {}
    //^^leave this constructor by the declarations

  public:
    template <typename... Args>
    void printx(const char* format, Args... args) const {
        char buffer[1024];
        snprintf(buffer, 1024, format, args...);
        localInteractionPtr->displayOutput(buffer);
    }

    template <typename... Args>
    void error(const char* format, Args... args) const {
        if (verbose >= 0) printx(format, args...);
    }

    template <typename... Args>
    void major_info(const char* format, Args... args) const {
        if (verbose >= 1) printx(format, args...);
    }

    template <typename... Args>
    void minor_info(const char* format, Args... args) const {
        if (verbose >= 2) printx(format, args...);
    }

    template <typename... Args>
    void extra_info(const char* format, Args... args) const {
        if (verbose >= 3) printx(format, args...);
    }

    template <typename... Args>
    void debug(const char* format, Args... args) const {
#ifdef CPPDEBUG
        if (verbose >= 4) printx(format, args...);
#endif
    }

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
}
