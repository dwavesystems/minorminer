#include "graph.hpp"
#include "gtest/gtest.h"

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

    queue.set_value(5, 0);

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

    queue.set_value(5, 10);
    queue.set_value(8, 6);

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
        queue.set_value(ii, 10 - ii);
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
        queue.set_value(ii, ii);
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

// Construct an empty queue
TEST(pairing_queue_fast_reset, construction) {
    pairing_queue::pairing_queue_fast_reset<int> queue(10);

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue
TEST(pairing_queue_fast_reset, construction_reset) {
    pairing_queue::pairing_queue_fast_reset<int> queue(10);
    queue.reset();

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(pairing_queue_fast_reset, insert_single) {
    pairing_queue::pairing_queue_fast_reset<int> queue(10);
    queue.reset();

    queue.set_value(5, 0);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 5);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());
}

TEST(pairing_queue_fast_reset, insert_two) {
    pairing_queue::pairing_queue_fast_reset<int> queue(10);
    queue.reset();

    queue.set_value(5, 10);
    queue.set_value(8, 6);

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
TEST(pairing_queue_fast_reset, insert_reverse) {
    pairing_queue::pairing_queue_fast_reset<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, 10 - ii);
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
TEST(pairing_queue_fast_reset, insert_forward) {
    pairing_queue::pairing_queue_fast_reset<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, ii);
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

// Construct an empty queue
TEST(pairing_queue_fast_reset_rtb, construction) {
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10);

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue
TEST(pairing_queue_fast_reset_rtb, construction_reset) {
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10);
    queue.reset();

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue
TEST(pairing_queue_fast_reset_rtb, construction_zrng_reset) {
    ZeroRNG zrng;
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10, zrng);

    int key, value;

    auto has_result = queue.pop_min(key, value);
    EXPECT_FALSE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(pairing_queue_fast_reset_rtb, insert_single) {
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10);
    queue.reset();

    queue.set_value(5, 0);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 5);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());
}

// Construct an empty queue and reset it to a given value
TEST(pairing_queue_fast_reset_rtb, insert_single_rng) {
    IncreasingRNG irng;
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10, irng);
    queue.reset();

    queue.set_value(5, 0);

    int key, value;
    auto has_result = queue.pop_min(key, value);
    EXPECT_EQ(value, 0);
    EXPECT_EQ(key, 5);
    EXPECT_TRUE(has_result);
    EXPECT_TRUE(queue.empty());
}

TEST(pairing_queue_fast_reset_rtb, insert_two) {
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10);
    queue.reset();

    queue.set_value(5, 10);
    queue.set_value(8, 6);

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

// Fill a queue with priorities reversed to their keys, and flush it
TEST(pairing_queue_fast_reset_rtb, insert_reverse) {
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, 10 - ii);
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
TEST(pairing_queue_fast_reset_rtb, insert_reverse_zrng) {
    ZeroRNG zrng;
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10, zrng);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, 10 - ii);
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

// Fill a queue with priorities reversed to their keys, and flush it (with a order-reversing tiebreaker)
TEST(pairing_queue_fast_reset_rtb, insert_reverse_drng) {
    IncreasingRNG irng;
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10, irng);
    queue.reset();

    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, 10 - ii);
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
TEST(pairing_queue_fast_reset_rtb, insert_forward_reorder_notb) {
    ZeroRNG zrng;
    IncreasingRNG irng;
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10);
    queue.reset();
    queue.reorder(zrng);
    int key, value;
    bool has_result;

    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, ii);
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
        queue.set_value(ii, ii);
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
TEST(pairing_queue_fast_reset_rtb, insert_forward_reorder_tb) {
    ZeroRNG zrng;
    IncreasingRNG irng;
    pairing_queue::pairing_queue_fast_reset_rtb<int> queue(10);
    queue.reset();
    queue.reorder(zrng);

    int key, value;
    bool has_result;

    for (int ii = 0; ii < 10; ii++) {
        queue.set_value(ii, 0);
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
        queue.set_value(ii, 0);
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
