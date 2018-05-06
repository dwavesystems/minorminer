#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

#include <iso646.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "debug.hpp"
#include "util.hpp"

// Macros local to this file, undefined at the end
#define max_P (numeric_limits<P>::max())

namespace pairing_queue {
// Import std library components
using std::vector;
using std::fill;
using std::numeric_limits;

template <typename N>
struct heap_link {
    N *next, *desc, *prev;
};

template <typename P>
struct value_field {
    P val;
    inline bool operator<(const value_field<P> &b) const { return (val < b.val) || ((val == b.val) && (this < &b)); }
};

template <typename P, typename K>
struct order_field : value_field<P> {
    using super = value_field<P>;
    K order;
    inline bool operator<(const order_field<P, K> &b) const {
        P v = static_cast<super>(b).val;
        return (super::val < v) || ((super::val == v) && (order < b.order));
    }
};

struct time_field {
    int time;
};

template <typename P>
struct plain_node : heap_link<plain_node<P>>, value_field<P> {};

template <typename P>
struct time_node : heap_link<time_node<P>>, value_field<P>, time_field {};

template <typename P, typename K>
struct order_node : heap_link<order_node<P, K>>, order_field<P, K>, time_field {};

//! A priority queue based on a pairing heap, with fixed memory footprint and support for a decrease-key operation
template <typename P, typename N = plain_node<P>>
class pairing_queue {
  public:
    typedef P value_type;

  protected:
    vector<N> nodes;

    N *root;

  public:
    pairing_queue(int n) : nodes(n), root(nullptr) {}

    //! swap the memory of self with another pairing_queue
    void swap(pairing_queue<P, N> &other) {
        nodes.swap(other.nodes);
        std::swap(root, other.root);
    }

    //! Reset the queue and fill the values with a default
    inline void reset_fill(const P v) {
        root = nullptr;
        for (auto &n : nodes) {
            reset(&n, v);
        }
    }

    //! Reset the queue and set the default to the maximum value
    inline void reset() { reset_fill(max_P); }

    //! Size of the queue
    inline int size() const { return nodes.size(); }

  protected:
    // blank out the links, except `prev`, which points back to n indicating
    // that this node is not currently in the queue
    inline void reset(N *n) {
        minorminer_assert(!empty(n));
        n->desc = nullptr;
        n->next = nullptr;
        n->prev = n;
    }

    //! reset the node `n` and set its value to `v`
    inline void reset(N *n, P v) {
        reset(n);
        n->val = v;
    }

  public:
    //! Remove the minimum value
    //! return true if any change is made
    inline bool delete_min() {
        if (empty()) return false;

        N *newroot = root->desc;
        if (!empty(newroot)) {
            newroot = merge_pairs(newroot);
            newroot->prev = nullptr;
            newroot->next = nullptr;
        }
        reset(root);
        root = newroot;
        return true;
    }

    //! Remove and return (in args) the minimum key, value pair
    inline bool pop_min(int &key, P &value) {
        if (empty()) {
            return false;
        }
        key = min_key();
        value = min_value();
        delete_min();
        return true;
    }

  public:
    //! Decrease the value of k to v
    //! NOTE: Assumes that v is lower than the current value of k
    inline void decrease_value(int k, const P &v) { decrease_value(node(k), v); }

  protected:
    //! protected variant of `decrease_value` using a node pointer
    inline void decrease_value(N *n, const P &v) {
        minorminer_assert(!empty(n));
        minorminer_assert(v < n->val);

        n->val = v;
        decrease(n);
    }

  public:
    //! Decrease the value of k to v
    //! Does nothing if v isn't actually a decrease.
    inline bool check_decrease_value(int k, const P &v) { return check_decrease_value(node(k), v); }

  protected:
    //! protected variant of `check_decrease_value` using a node pointer
    inline bool check_decrease_value(N *n, const P &v) {
        minorminer_assert(!empty(n));
        if (v < n->val) {
            n->val = v;
            decrease(n);
            return true;
        } else {
            return false;
        }
    }

  public:
    //! set the value associated with `k` to `v`
    inline void set_value(int k, const P &v) { set_value(node(k), v); }

  protected:
    //! protected variant of `set_value` using a node pointer
    inline void set_value(N *n, const P &v) {
        minorminer_assert(!empty(n));
        if (n->prev == n) {
            n->val = v;
            root = merge_roots(n, root);
        } else if (v < n->val) {
            n->val = v;
            decrease(n);
        } else if (n->val < v) {
            n->val = v;
            remove(n);
            root = merge_roots(n, root);
        }
    }

  public:
    //! set the value associated with `k` to `v`, without making any other modifications
    //! to the internal data structure
    inline void set_value_unsafe(int k, const P &v) { set_value_unsafe(node(k), v); }

  protected:
    // protected variant of `set_value_unsafe` using a node pointer
    inline void set_value_unsafe(N *n, const P &v) {
        minorminer_assert(!empty(n));
        n->val = v;
    }

  public:
    //! get the current minimum value (assumes `!empty()`)
    inline P min_value() const {
        minorminer_assert(!empty());
        return root->val;
    }

    //! get the current minimum-value key (assumes `!empty()`)
    inline int min_key() const {
        minorminer_assert(!empty());
        return key(root);
    }

  protected:
    //! get the key of a node
    inline int key(N *n) const { return n - nodes.data(); }

  public:
    //! trueue this queue is empty
    inline bool empty(void) const { return empty(root); }

  protected:
    //! protected variant of `empty`, can be used for non-root nodes
    inline bool empty(N *n) const { return n == nullptr; }

  public:
    //! return the stored value for `k`.  does not check that this value has
    //! been initialized or re-initialized since the last reset
    inline P value(int k) const { return const_node(k)->val; }

  protected:
    //! node pointer accessor
    inline N *node(int k) {
        minorminer_assert(0 <= k && k < size());
        return nodes.data() + k;
    }

    //! const node pointer accessor
    inline const N *const_node(int k) const {
        minorminer_assert(0 <= k && k < size());
        return nodes.data() + k;
    }

    //! INTERNAL USE ONLY most general merge_roots function.  assumes that
    //! `a` is not null
    inline N *merge_roots(N *a, N *b) {
        // even this version of merge_roots is slightly unsafe -- we never call it with a null, so let's not check!
        // * doesn't check for nullval
        minorminer_assert(!empty(a));

        if (empty(b)) return a;
        N *c = merge_roots_unsafe(a, b);
        c->prev = nullptr;
        return c;
    }

    //! INTERNAL USE ONLY merge_roots, assuming both `a` and `b` are not
    //! null, possibly invalidating the internal data structure (see source
    //! for details)
    inline N *merge_roots_unsafe(N *a, N *b) {
        // this unsafe version of merge_roots which
        // * doesn't check for nullval
        // * doesn't ensure that the returned node has prev[a] = nullval
        minorminer_assert(!empty(a));
        minorminer_assert(!empty(b));

        if (*a < *b)
            return merge_roots_unchecked(a, b);
        else
            return merge_roots_unchecked(b, a);
    }

    //! INTERNAL USE ONLY merge_roots, assuming both `a` and `b` are not
    //! null, and that `value(a)` < `value(b)`.  may invalidate the internal
    //! data structure (see source for details)
    inline N *merge_roots_unchecked(N *a, N *b) {
        // this very unsafe version of self.merge_roots which
        // * doesn't check for nullval
        // * doesn't ensure that the returned node has prev[a] = nullval
        // * doesn't check that a < b
        minorminer_assert(!empty(a));
        minorminer_assert(!empty(b));
        // minorminer_assert(a < b);

        N *c = b->next = a->desc;
        if (!empty(c)) c->prev = b;
        b->prev = a;
        a->desc = b;
        return a;
    }

    //! INTERNAL USE ONLY merge_pairs is the "magic" behind the pairing queue.
    //! when the queue pops, we must maintain the condition that
    //! `root->next = nullptr`.  we traverse the `newroot->next->(next->)*`
    //! linked list twice, first merging subsequent pairs, to produce a new
    //! linked list of half the size, and then merging the head of that list
    //! with its next until we're down to a single node
    inline N *merge_pairs(N *a) {
        if (empty(a)) return nullptr;
        N *r = nullptr;
        do {
            N *b = a->next;
            if (!empty(b)) {
                N *c = b->next;
                b = merge_roots_unsafe(a, b);
                b->prev = r;
                r = b;
                a = c;
            } else {
                a->prev = r;
                r = a;
                break;
            }
        } while (!empty(a));
        a = r;
        r = a->prev;
        while (!empty(r)) {
            N *t = r->prev;
            a = merge_roots_unsafe(a, r);
            r = t;
        }
        return a;
    }

    //! INTERNAL USE ONLY removes node `a` from the pairing queue, assuming
    //! it is not the root
    inline void remove(N *a) {
        minorminer_assert(!empty(a));
        N *b = a->prev;
        N *c = a->next;
        minorminer_assert(!empty(b));
        if (b->desc == a)
            b->desc = c;
        else
            b->next = c;

        if (!empty(c)) {
            c->prev = b;
            a->next = nullptr;
        }
    }

    //! update the data structure to reflect a decrease in the value of `a`
    inline void decrease(N *a) {
        minorminer_assert(!empty(a));
        if (!empty(a->prev)) {
            minorminer_assert(a != root);  // theoretically, root is the only node with empty(prev)
            remove(a);
            root = merge_roots(a, root);
        }
    }
};

//! This is a specialization of the pairing_queue that has a constant time
//! reset method, at the expense of an extra check when values are set or updated.
template <typename P, typename N = time_node<P>>
class pairing_queue_fast_reset : public pairing_queue<P, N> {
    using super = pairing_queue<P, N>;

  protected:
    int now;

    //! reset the node `n` and make it current
    inline void reset(N *n) {
        super::reset(n);
        n->time = now;
    }

    //! check if the node `n` is current (has `time=now`) and if not,
    //! reset it (making it current)
    inline bool current(N *n) {
        if (n->time != now) {
            reset(n);
            return false;
        }
        return true;
    }

  public:
    pairing_queue_fast_reset(int n) : super(n), now(0) {}

    //! swap the memory of self with another pairing_queue_fast_reset
    void swap(pairing_queue_fast_reset<P, N> &other) {
        super::swap(other);
        std::swap(now, other.now);
    }

    //! clear out this data structure or make it ready for the first time
    inline void reset() {
        super::root = nullptr;
        if (!now++) {
            for (auto &n : super::nodes) n.time = 0;
        }
    }

    //! set the value associated with `k` to `v`, without making any other modifications
    //! to the internal data structure
    inline void set_value_unsafe(int k, const P &v) {
        auto n = super::node(k);
        current(n);
        super::set_value_unsafe(n, v);
    }

    //! set the value associated with `k` to `v`
    inline void set_value(int k, const P &v) {
        auto n = super::node(k);
        if (current(n))
            super::set_value(n, v);
        else {
            n->val = v;
            super::root = super::merge_roots(n, super::root);
        }
    }

    //! Decrease the value of k to v
    //! Does nothing if v isn't actually a decrease.
    inline bool check_decrease_value(int k, const P &v) {
        auto n = super::node(k);
        if (current(n))
            return super::check_decrease_value(n, v);
        else {
            super::set_value(n, v);
            return true;
        }
    }

    //! Safe value getter.  If `k` doesn't have a value (safely or unsafely set)
    //! since the last reset, returns numeric_limits<P>::max().  This works even
    //! after `k` has been popped.
    inline P get_value(int k) const {
        auto n = super::const_node(k);
        if (n->time == now)
            return n->val;
        else
            return max_P;
    }
};

//! This is a specialization of the pairing_queue_fast_reset which implements
//! random tie-breaking.  the random tie-breaking is based on randomized
//! fields which are not changed on `reset()`.  that means that the randomization
//! is deterministic for the sake of using this as a priority queue, but
//! only properly "random" if a `reorder` is called with every `reset`.
//! for our purposes, it is sufficient (and somewhat preferable) to initialize
//! a number of these queues with an RNG and occasionally/randomly swap which
//! queue is used for what
template <typename P, typename K = uint64_t, typename N = order_node<P, K>>
class pairing_queue_fast_reset_rtb : public pairing_queue_fast_reset<P, N> {
    using super = pairing_queue_fast_reset<P, N>;
    using grand = pairing_queue<P, N>;

  public:
    //! this constructor calls `rng()` for each of `n` nodes,
    //! storing a value for tie-breaking purposes.
    template <typename R>
    pairing_queue_fast_reset_rtb(int n, R &rng) : super(n) {
        reorder(rng);
    }

    //! this constructor is generally bad idea, but tie-breaking is still
    //! deterministic
    pairing_queue_fast_reset_rtb(int n) : super(n) {
        int k = 0;
        for (auto &n : grand::nodes) n.order = k++;
    }

  protected:
    //! updates the tie-breaker for `n`
    template <typename R>
    inline void reorder(N *n, R &rng, int size, int ord) {
        n->order = rng() * size + ord;
    }

    //! fetch the tie-breaker for `n`
    inline K get_order(int k) const { return grand::const_node(k)->order; }

  public:
    //! refresh the tie-breaking with a new set of random values
    template <typename R>
    inline void reorder(R &rng) {
        int size = grand::nodes.size();
        for (int k = size; k--;) {
            reorder(grand::node(k), rng, size, k);
        }
    }

    //! duplicate the tie-breaking of another pairing_queue_fast_reset_rtb
    inline void reorder_copy(const pairing_queue_fast_reset_rtb<P, K, N> &other) {
        int size = grand::nodes.size();
        for (int k = size; k--;) {
            grand::node(k)->order = other.get_order(k);
        }
    }
};

#undef nullval
#undef max_P
}
