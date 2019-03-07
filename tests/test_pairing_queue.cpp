#include <random>
#include "debug.hpp"
#include "gtest/gtest.h"
#include "pairing_queue.hpp"

using namespace find_embedding;
using distance_queue = pairing_queue<priority_node<int, min_heap_tag>>;

struct node_t {
    int x;
    node_t(int y) : x(y) {}
    node_t() {}
    bool operator<(const node_t &b) const { return x > b.x; }
};

// Construct an empty queue
TEST(pairing_queue, construction) {
    pairing_queue<node_t> queue(10);

    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and insert a single value
TEST(pairing_queue, insert_single) {
    pairing_queue<node_t> queue(10);
    queue.reset();

    queue.emplace(5);

    auto value = queue.top();
    EXPECT_EQ(value.x, 5);

    queue.pop();
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue, fill with data in reverse
TEST(pairing_queue, insert_reverse) {
    pairing_queue<node_t> queue(10);
    queue.reset();

    for (int ii = 10; ii--;) {
        queue.emplace(ii);
    }

    for (int ii = 0; ii < 10; ii++) {
        auto value = queue.top();
        EXPECT_EQ(value.x, ii);
        EXPECT_FALSE(queue.empty());
        queue.pop();
    }

    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue, fill with data in order
TEST(pairing_queue, insert_forward) {
    pairing_queue<node_t> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.emplace(ii);
    }

    for (int ii = 0; ii < 10; ii++) {
        auto value = queue.top();
        EXPECT_EQ(value.x, ii);
        EXPECT_FALSE(queue.empty());
        queue.pop();
    }

    EXPECT_TRUE(queue.empty());
}

// Fill a queue with priorities reversed to their keys, and flush it (with an order-preserving tiebreaker)
TEST(pairing_queue, insert_reverse_ftb) {
    distance_queue queue(10);

    for (int ii = 0; ii < 10; ii++) {
        queue.emplace(ii, ii, 10 - ii);
    }

    for (int ii = 0; ii < 10; ii++) {
        auto value = queue.top();
        EXPECT_EQ(value.dist, ii + 1);
        EXPECT_EQ(value.node, 9 - ii);
        EXPECT_FALSE(queue.empty());
        queue.pop();
    }

    EXPECT_TRUE(queue.empty());
}

// Fill a queue with priorities equal to keys, and flush it -- (with an order-preserving tiebreaker)
TEST(pairing_queue, insert_forward_ftb) {
    distance_queue queue(10);

    for (int ii = 0; ii < 10; ii++) {
        queue.emplace(ii, ii, ii);
    }

    for (int ii = 0; ii < 10; ii++) {
        auto value = queue.top();
        EXPECT_EQ(value.dist, ii);
        EXPECT_EQ(value.node, ii);
        EXPECT_FALSE(queue.empty());
        queue.pop();
    }

    EXPECT_TRUE(queue.empty());
}

// Repeatedly fill a queue with randomized dists & dirts
TEST(pairing_queue, insert_random) {
    distance_queue queue(100);
    std::default_random_engine rng;
    std::vector<int> dirt(100);
    std::vector<int> dist(100);
    for (int ii = 100; ii--;) {
        dirt[ii] = ii;
        dist[ii] = ii / 4;
    }

    for (int ii = 100; ii--;) {
        std::shuffle(dirt.begin(), dirt.end(), rng);
        std::shuffle(dist.begin(), dist.end(), rng);
        for (int jj = 100; jj--;) queue.emplace(jj, dirt[jj], dist[jj]);

        int last_dirt = -1;
        int last_dist = -1;
        for (int jj = 100; jj--;) {
            auto value = queue.top();
            EXPECT_TRUE((last_dist < value.dist) || ((last_dist == value.dist) && (last_dirt < value.dirt)));
            last_dist = value.dist;
            last_dirt = value.dirt;
            EXPECT_FALSE(queue.empty());
            queue.pop();
        }
        queue.reset();
        EXPECT_TRUE(queue.empty());
    }
}
