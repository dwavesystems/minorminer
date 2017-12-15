#include <vector>
#include "chain.hpp"
#include "gtest/gtest.h"
#include "util.hpp"
using std::vector;

struct embedding {
    std::vector<int> qubit_weights;
    std::vector<find_embedding::chain> var_embedding;
    embedding(int num_qubits, int num_vars) : qubit_weights(num_qubits, 0) {
        for (int v = 0; v < num_vars; v++) var_embedding.emplace_back(qubit_weights, v);
    }
};

class embedding_problem_t {
  public:
    inline bool accepts_qubit(int v, int q) { return true; }
};

//
TEST(chain, construction_empty) {
    std::mt19937_64 rng(0);
    std::vector<int> weight(5, 0);
    find_embedding::chain c(weight, 0);
    ASSERT_EQ(c.run_diagnostic(), 0);
}

TEST(chain, construction_root) {
    std::mt19937_64 rng(0);
    std::vector<int> weight(5, 0);
    find_embedding::chain c(weight, 0);
    c.set_root(0);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 1);
}

TEST(chain, trim_root_bounce) {
    std::mt19937_64 rng(0);
    std::vector<int> weight(5, 0);
    find_embedding::chain c(weight, 0);
    c.set_root(0);
    c.trim_leaf(0);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 1);
}

TEST(chain, trim_root_branch_bounce) {
    std::mt19937_64 rng(0);
    std::vector<int> weight(5, 0);
    find_embedding::chain c(weight, 0);
    c.set_root(0);
    c.trim_branch(0);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 1);
}

TEST(chain, add_leaves_path) {
    std::mt19937_64 rng(0);
    std::vector<int> weight(5, 0);
    find_embedding::chain c(weight, 0);
    c.set_root(0);
    c.add_leaf(1, 0);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 2);
    c.add_leaf(2, 1);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 3);
    c.add_leaf(3, 2);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 4);
    c.add_leaf(4, 3);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 5);
}

TEST(chain, trim_branch) {
    std::mt19937_64 rng(0);
    std::vector<int> weight(5, 0);
    find_embedding::chain c(weight, 0);
    c.set_root(0);
    c.add_leaf(1, 0);
    c.add_leaf(2, 1);
    c.add_leaf(3, 2);
    c.add_leaf(4, 3);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 5);
    c.trim_branch(4);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(c.size(), 1);
}

TEST(chain, linkpath) {
    std::mt19937_64 rng(0);
    std::vector<int> weight(50, 0);
    find_embedding::chain c(weight, 0);
    find_embedding::chain d(weight, 1);
    find_embedding::chain e(weight, 2);
    std::vector<int> parents(50, -1);
    c.set_root(0);

    d.set_root(5);
    parents[0] = 1;
    parents[1] = 2;
    parents[2] = 3;
    parents[3] = 5;
    c.link_path(d, 0, parents);

    e.set_root(10);
    parents[0] = 2;
    parents[2] = 1;
    parents[1] = 3;
    parents[3] = 4;
    parents[4] = 10;
    c.link_path(e, 0, parents);

    ASSERT_EQ(c.size(), 5);
    ASSERT_EQ(d.size(), 1);
    ASSERT_EQ(e.size(), 1);
}

TEST(chain, linkpathandsteal) {
    embedding_problem_t mock;
    std::mt19937_64 rng(0);
    std::vector<int> weight(50, 0);
    find_embedding::chain c(weight, 0);
    find_embedding::chain d(weight, 1);
    find_embedding::chain e(weight, 2);
    std::vector<int> parents(50, -1);
    c.set_root(0);

    d.set_root(5);
    parents[0] = 1;
    parents[1] = 2;
    parents[2] = 3;
    parents[3] = 5;
    c.link_path(d, 0, parents);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(d.run_diagnostic(), 0);

    e.set_root(10);
    parents[0] = 2;
    parents[2] = 1;
    parents[1] = 3;
    parents[3] = 4;
    parents[4] = 10;  // to be stolen
    c.link_path(e, 0, parents);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(e.run_diagnostic(), 0);

    ASSERT_EQ(c.size(), 5);
    ASSERT_EQ(d.size(), 1);
    ASSERT_EQ(e.size(), 1);

    d.steal(c, mock, 0);
    e.steal(c, mock, 0);
    ASSERT_EQ(c.run_diagnostic(), 0);
    ASSERT_EQ(d.run_diagnostic(), 0);
    ASSERT_EQ(e.run_diagnostic(), 0);

    ASSERT_EQ(c.size(), 4);
    ASSERT_EQ(d.size(), 1);
    ASSERT_EQ(e.size(), 2);
}

TEST(chain, balancechains) {
    embedding_problem_t mock;
    std::mt19937_64 rng(0);
    std::vector<int> weight(50, 0);
    find_embedding::chain c(weight, 0);
    find_embedding::chain d(weight, 1);
    std::vector<int> parents(50, -1);
    for (int i = 0; i < 50; i++) parents[i] = i - 1;
    c.set_root(0);
    d.set_root(49);
    d.link_path(c, 49, parents);

    c.steal(d, mock, 25);

    ASSERT_EQ(c.size(), 25);
    ASSERT_EQ(d.size(), 25);
}
