#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "pairing_queue.hpp"
using std::vector;

// Construct an empty queue
TEST(pairing_queue, construction) {
    pairing_queue::pairing_queue<int> queue(10);

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue
TEST(pairing_queue, construction_reset) {
    pairing_queue::pairing_queue<int> queue(10);
    queue.reset();

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(pairing_queue, insert_single) {
    pairing_queue::pairing_queue<int> queue(10);
    queue.reset();

    queue.check_insert(5, 0);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 5);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());
}

TEST(pairing_queue, insert_two) {
    pairing_queue::pairing_queue<int> queue(10);
    queue.reset();

    queue.check_insert(5, 10);
    queue.check_insert(8, 6);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    ASSERT_TRUE(has_result);
    EXPECT_EQ(value, 6);
    EXPECT_EQ(key, 8);
    ASSERT_FALSE(queue.empty());

    has_result = queue.pop_min(key, value);
    ASSERT_TRUE(has_result);
    EXPECT_EQ(value, 10);
    EXPECT_EQ(key, 5);
    ASSERT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(pairing_queue, insert_reverse) {
    pairing_queue::pairing_queue<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, 10 - ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii + 1);
        EXPECT_EQ(key, 9 - ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 0);
    EXPECT_EQ(value, 10);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(pairing_queue, insert_forward) {
    pairing_queue::pairing_queue<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii);
        EXPECT_EQ(key, ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 9);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

class IncreasingRNG {
    uint64_t n = std::numeric_limits<uint64_t>::min();

  public:
    IncreasingRNG() {}
    uint64_t operator()() { return n++; }
};

class ZeroRNG {
  public:
    ZeroRNG() {}
    uint64_t operator()() { return 0ULL; }
};

// Construct an empty queue with an RNG
TEST(pairing_queue, construction_zrng_reset) {
    ZeroRNG zrng;
    pairing_queue::pairing_queue<int> queue(10, zrng);

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue, add two tied elements and check that they come out in the proper order
TEST(pairing_queue, insert_tiebreak) {
    IncreasingRNG irng;
    ZeroRNG zrng;
    pairing_queue::pairing_queue<int> queue(10, irng);
    queue.reset();

    queue.check_insert(5, 0);
    queue.check_insert(6, 0);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 6);
    EXPECT_TRUE(has_result);
    has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 5);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    queue.reorder(zrng);

    queue.reset();

    queue.check_insert(5, 0);
    queue.check_insert(6, 0);

    has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 5);
    EXPECT_TRUE(has_result);
    has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 6);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue with priorities reversed to their keys, and flush it (with an order-preserving tiebreaker)
TEST(pairing_queue, insert_reverse_zrng) {
    ZeroRNG zrng;
    pairing_queue::pairing_queue<int> queue(10, zrng);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, 10 - ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii + 1);
        EXPECT_EQ(key, 9 - ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 0);
    EXPECT_EQ(value, 10);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue with priorities reversed to their keys, and flush it (with an order-preserving tiebreaker)
TEST(pairing_queue, insert_reverse_drng) {
    IncreasingRNG irng;
    pairing_queue::pairing_queue<int> queue(10, irng);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, 10 - ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii + 1);
        EXPECT_EQ(key, 9 - ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 0);
    EXPECT_EQ(value, 10);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue with priorities equal to keys, and flush it -- testing resets by different RNGs (shouldn't impact
// anything)
TEST(pairing_queue, insert_forward_reorder_notb) {
    ZeroRNG zrng;
    IncreasingRNG irng;
    pairing_queue::pairing_queue<int> queue(10);
    queue.reset();
    queue.reorder(zrng);
    int key, value;
    bool has_result;

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii);
        EXPECT_EQ(key, ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 9);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());

    queue.reset();
    queue.reorder(irng);

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii);
        EXPECT_EQ(key, ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 9);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue with all priorities zero, and flush it -- first with an order-preserving tiebreaker and then
// order-reversing (should reverse order)
TEST(pairing_queue, insert_forward_reorder_tb) {
    ZeroRNG zrng;
    IncreasingRNG irng;
    pairing_queue::pairing_queue<int> queue(10);
    queue.reset();
    queue.reorder(zrng);

    int key, value;
    bool has_result;

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, 0);
    }

    for (int ii = 0; ii < 9; ii++) {
        has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, 0);
        EXPECT_EQ(key, ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 0);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());

    queue.reset();
    queue.reorder(irng);

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, 0);
    }

    for (int ii = 0; ii < 9; ii++) {
        has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, 0);
        EXPECT_EQ(key, 9 - ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 0);
    EXPECT_EQ(value, 0);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue
TEST(decrease_queue, construction) {
    pairing_queue::decrease_queue<int> queue(10);

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue
TEST(decrease_queue, construction_reset) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(decrease_queue, insert_single) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    queue.check_insert(5, 0);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 5);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());
}

TEST(decrease_queue, insert_two) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    queue.check_insert(5, 10);
    queue.check_insert(8, 6);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    ASSERT_TRUE(has_result);
    EXPECT_EQ(value, 6);
    EXPECT_EQ(key, 8);
    ASSERT_FALSE(queue.empty());

    has_result = queue.pop_min(key, value);
    ASSERT_TRUE(has_result);
    EXPECT_EQ(value, 10);
    EXPECT_EQ(key, 5);
    ASSERT_TRUE(queue.empty());
}

// Fill a queue with random values, ensure sortedness.
TEST(pairing_queue, checkinsert_vs_sort) {
    std::random_device rng;
    vector<unsigned int> values(10);
    pairing_queue::pairing_queue<unsigned int> queue(10);
    queue.reset();

    for (int i = 0; i < 500; i++) {
        unsigned int v = rng();
        int k = i % 10;
        if (queue.check_insert(k, v)) values[k] = v;
    }

    vector<int> keys;
    unsigned int last_v;
    for (int i = 0; i < 10; i++) {
        int k = queue.min_key();
        unsigned int v = queue.min_value();
        queue.delete_min();
        keys.push_back(k);
        EXPECT_EQ(v, values[k]);
        if (i) EXPECT_LE(last_v, v);
        last_v = v;
    }
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(decrease_queue, insert_reverse) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, 10 - ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii + 1);
        EXPECT_EQ(key, 9 - ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 0);
    EXPECT_EQ(value, 10);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(decrease_queue, insert_forward) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii);
        EXPECT_EQ(key, ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 9);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue, decrease all of its values
TEST(decrease_queue, check_decrease) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, ii);
    }
    for (int ii = 0; ii < 10; ii++) {
        queue.check_decrease_value(ii, ii / 2);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(key, ii);
        EXPECT_EQ(value, ii / 2);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 9 / 2);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue, and fail to decrease any of its values
TEST(decrease_queue, check_decrease2) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, ii);
    }
    for (int ii = 0; ii < 10; ii++) {
        queue.check_decrease_value(ii, 2 * ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(value, ii);
        EXPECT_EQ(key, ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 9);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue, and increase all of its values
TEST(decrease_queue, increase) {
    pairing_queue::decrease_queue<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.check_insert(ii, ii);
    }
    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, 2 * ii);
    }

    for (int ii = 0; ii < 9; ii++) {
        int key, value;
        auto has_result = queue.pop_min(key, value);
        EXPECT_EQ(key, ii);
        EXPECT_EQ(value, 2 * ii);
        EXPECT_TRUE(has_result);
        EXPECT_FALSE(queue.empty());
    }

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(key, 9);
    EXPECT_EQ(value, 18);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());

    has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Fill a queue with random values, ensure sortedness.
TEST(decrease_queue, setvalue_vs_sort) {
    std::random_device rng;
    vector<unsigned int> values(10);
    pairing_queue::decrease_queue<unsigned int> queue(10);
    queue.reset();

    for (int i = 0; i < 500; i++) {
        unsigned int v = rng();
        int k = i % 10;
        queue.set_value(k, v);
        values[k] = v;
    }

    vector<int> keys;
    unsigned int last_v;
    for (int i = 0; i < 10; i++) {
        int k = queue.min_key();
        unsigned int v = queue.min_value();
        queue.delete_min();
        keys.push_back(k);
        EXPECT_EQ(v, values[k]);
        if (i) EXPECT_LE(last_v, v);
        last_v = v;
    }
    EXPECT_TRUE(queue.empty());
}

// Fill a queue with random values, ensure sortedness.
TEST(decrease_queue, checkdecrease_vs_sort) {
    std::random_device rng;
    vector<unsigned int> values(10);
    pairing_queue::decrease_queue<unsigned int> queue(10);
    queue.reset();

    for (int i = 0; i < 500; i++) {
        unsigned int v = rng();
        int k = i % 10;
        if (queue.check_decrease_value(k, v)) values[k] = v;
    }

    vector<int> keys;
    unsigned int last_v;
    for (int i = 0; i < 10; i++) {
        int k = queue.min_key();
        unsigned int v = queue.min_value();
        queue.delete_min();
        keys.push_back(k);
        EXPECT_EQ(v, values[k]);
        if (i) EXPECT_LE(last_v, v);
        last_v = v;
    }
    EXPECT_TRUE(queue.empty());
}
