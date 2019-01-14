#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

#include <iso646.h>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "debug.hpp"

// Macros local to this file, undefined at the end
#define max_P (numeric_limits<P>::max())

namespace pairing_queue {
// Import std library components
using std::numeric_limits;
typedef time_t uint64_t;

//! merge_pairs is the "magic" behind the pairing queue.
//! when the queue pops, we must maintain the condition that
//! `root->next = nullptr`.  we traverse the `newroot->next->(next->)*`
//! linked list twice, first merging subsequent pairs, to produce a new
//! linked list of half the size, and then merging the head of that list
//! with its next until we're down to a single node
template <typename N>
inline N *_merge_pairs(N *a) {
    N *r = nullptr;
    do {
        N *b = a->next;
        if (b != nullptr) {
            N *c = b->next;
            b = _merge_roots_unsafe(a, b);
            b->next = r;
            r = b;
            a = c;
        } else {
            a->next = r;
            r = a;
            break;
        }
    } while (a != nullptr);
    a = r;
    r = a->next;
    while (r != nullptr) {
        N *t = r->next;
        a = _merge_roots_unsafe(a, r);
        r = t;
    }
    return a;
}

//! merge_roots, assuming `other` is not null null, possibly invalidating
//! the internal data structure (see source for details)
template <typename N>
inline N *_merge_roots_unsafe(N *a, N *b) {
    // this unsafe version of merge_roots which
    // * doesn't check for nullval
    // * doesn't ensure that the returned node has next = nullval
    minorminer_assert(a != nullptr);
    minorminer_assert(b != nullptr);

    if (*a < *b)
        return a->merge_roots_unchecked(b);
    else
        return b->merge_roots_unchecked(a);
}

template <typename P, typename K>
struct order_node {
    time_t time;
    K order;
    order_node<P, K> *next, *desc;
    P val;
    inline bool operator<(const order_node<P, K> &other) const {
        return (val < other.val) || ((val == other.val) && (order < other.order));
    }

    // initialize this node and update its timestamp
    inline void reset(time_t now) {
        desc = nullptr;
        next = this;
        time = now;
    }

    // initialize this node
    inline void reset() {
        desc = nullptr;
        next = this;
    }

    inline bool active() { return next != this; }

    inline order_node<P, K> *merge_pairs() { return _merge_pairs(this); }

    //! the basic operation of the pairing queue -- put `this` and `other`
    //! into heap-order
    inline order_node<P, K> *merge_roots(order_node<P, K> *other) {
        if (other == nullptr) return this;
        order_node<P, K> *c = _merge_roots_unsafe(this, other);
        c->next = nullptr;
        return c;
    }

    //! merge_roots, assuming `other` is not null and that `val` < `other->val`.
    //!  may invalidate the internal data structure (see source for details)
    inline order_node<P, K> *merge_roots_unchecked(order_node<P, K> *other) {
        // this very unsafe version of self.merge_roots which
        // * doesn't check for nullval
        // * doesn't ensure that the returned node has next = nullval
        // * doesn't check that this < other
        minorminer_assert(other != nullptr);
        minorminer_assert(*this < *other);

        other->next = desc;
        desc = other;
        return this;
    }
};

template <typename P>
struct key_node {
    P val;
    time_t time;
    key_node<P> *next, *desc, *prev;
    inline bool operator<(const key_node<P> &other) const {
        return (val < other.val) || ((val == other.val) && (this < &other));
    }

    // initialize this node and update its timestamp
    inline void reset(time_t now) {
        reset();
        time = now;
    }

    // initialize this node
    inline void reset() {
        next = desc = nullptr;
        prev = this;
    }

    inline bool active() { return next != this; }

    inline key_node<P> *merge_pairs() {
        key_node<P> *t = _merge_pairs(this);
        t->prev = nullptr;
        return t;
    }

    //! the basic operation of the pairing queue -- put `this` and `other`
    //! into heap-order
    inline key_node<P> *merge_roots(key_node<P> *other) {
        if (other == nullptr) return this;
        key_node<P> *c = _merge_roots_unsafe(this, other);
        c->next = nullptr;
        c->prev = nullptr;
        return c;
    }

    //! merge_roots, assuming `other` is not null and that `val` < `other->val`.
    //!  may invalidate the internal data structure (see source for details)
    inline key_node<P> *merge_roots_unchecked(key_node<P> *other) {
        // this very unsafe version of self.merge_roots which
        // * doesn't check for nullval
        // * doesn't ensure that the returned node has next = nullval
        // * doesn't check that this < other
        minorminer_assert(other != nullptr);
        minorminer_assert(*this < *other);

        key_node<P> *c = other->next = desc;
        if (c != nullptr) c->prev = other;
        other->prev = this;
        desc = other;
        return this;
    }

    // restructure after an increase-key, return a root
    inline key_node<P> *increase_root() {
        minorminer_assert(prev == nullptr);
        if (desc == nullptr || *this < *desc) {
            return this;
        } else {
            key_node<P> *d = desc->merge_pairs();
            desc = nullptr;
            return merge_roots(d);
        }
    }

    //! restructure after changing the priority of an internal node, leaving its
    //! neighbors in a sane state and making it a proper root
    inline void extract_root() {
        minorminer_assert(prev != nullptr);
        if (prev->desc == this)
            prev->desc = next;
        else
            prev->next = next;

        if (next != nullptr) {
            next->prev = prev;
            next = nullptr;
        }
        prev = nullptr;
    }
};

//! A priority queue based on a pairing heap, with fixed memory footprint and support for a decrease-key operation
template <typename P, typename N>
class base_queue {
    using self = base_queue<P, N>;

  public:
    typedef P value_type;

  protected:
    std::vector<N> nodes;

    N *root;

    time_t now;

  public:
    //-------------
    // constructors
    //-------------
    base_queue(int n) : nodes(n), root(nullptr) {}

    //-----------------------------------
    // priority-queue interface functions
    //-----------------------------------

    //! swap the memory of self with another base_queue
    void swap(base_queue<P, N> &other) {
        nodes.swap(other.nodes);
        std::swap(root, other.root);
        std::swap(now, other.now);
    }

    //! Size of the queue
    inline int size() const { return nodes.size(); }

    //! clear out this data structure or make it ready for the first time
    inline void reset() {
        root = nullptr;
        if (!now++) {
            for (auto &n : nodes) n.time = 0;
        }
    }

  protected:
    //! check if the node `n` is current (has `time=now`) and if not,
    //! reset_node it (making it current)
    inline bool current(N *n) {
        if (n->time != now) {
            n->reset(now);
            return false;
        }
        return true;
    }

  public:
    //! Remove the minimum value
    //! return true if any change is made
    inline bool delete_min() {
        if (empty()) return false;

        restructure_pop();
        return true;
    }

    //! Remove and return (in args) the minimum key, value pair
    inline bool pop_min(int &key, P &value) {
        if (empty()) return false;
        key = min_key();
        value = min_value();
        restructure_pop();
        return true;
    }

  protected:
    //! Remove the root and restructure.
    inline void restructure_pop() {
        N *newroot = root->desc;
        if (newroot != nullptr) newroot = newroot->merge_pairs();
        root->reset();
        root = newroot;
    }

  public:
    //! Set the value of `k` to `v`.  Does nothing if `k` has been inserted since the last reset.
    inline bool check_insert(int k, const P &v) { return check_insert(self::node(k), v); }

  protected:
    inline bool check_insert(N *n, const P &v) {
        minorminer_assert(n != nullptr);
        if (!current(n)) {
            n->val = v;
            root = n->merge_roots(root);
            return true;
        } else {
            return false;
        }
    }

  public:
    //! Safe value getter.  If `k` doesn't have a value (safely or unsafely set)
    //! since the last reset_node, returns numeric_limits<P>::max().  This works even
    //! after `k` has been popped.
    inline P get_value(int k) const { return get_value(const_node(k)); }

  protected:
    inline P get_value(const N *n) const {
        if (n->time == now)
            return n->val;
        else
            return max_P;
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
    //! been initialized or re-initialized since the last reset_node
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
};

//! A priority queue based on a pairing heap, with fixed memory footprint
template <typename P, typename K = uint64_t, typename N = order_node<P, K>>
class pairing_queue : public base_queue<P, N> {
    using super = base_queue<P, N>;
    friend class base_queue<P, N>;

  public:
    typedef P value_type;

    //-------------
    // constructors
    //-------------
    //! this constructor calls `rng()` for each of `n` nodes,
    //! storing a value for tie-breaking purposes.
    template <typename R>
    pairing_queue(int n, R &rng) : super(n) {
        reorder(rng);
    }

    //! this constructor is generally bad idea, but tie-breaking is still
    //! deterministic
    pairing_queue(int n) : super(n) {
        int k = 0;
        for (auto &n : super::nodes) n.order = k++;
    }

    //----------------------
    // tie-breaker functions
    //----------------------
  protected:
    //! updates the tie-breaker for `n`
    template <typename R>
    inline void reorder(N *n, R &rng, int size, int ord) {
        n->order = rng() * size + ord;
    }

    //! fetch the tie-breaker for `n`
    inline K get_order(int k) const { return super::const_node(k)->order; }

  public:
    //! refresh the tie-breaking with a new set of random values
    template <typename R>
    inline void reorder(R &rng) {
        int size = super::nodes.size();
        for (int k = size; k--;) {
            reorder(super::node(k), rng, size, k);
        }
    }

    //! duplicate the tie-breaking of another pairing_queue
    inline void reorder_copy(const pairing_queue<P, K, N> &other) {
        int size = super::nodes.size();
        for (int k = size; k--;) {
            super::node(k)->order = other.get_order(k);
        }
    }
};

//! A priority queue based on a pairing heap, with fixed memory footprint and support for a decrease-key operation
template <typename P, typename N = key_node<P>>
class decrease_queue : public base_queue<P, N> {
    using super = base_queue<P, N>;

  public:
    typedef P value_type;

    //-------------
    // constructors
    //-------------
    decrease_queue(int n) : super(n) {}

    //-----------------------------------
    // priority-queue interface functions
    //-----------------------------------

  public:
    //! Decrease the value of k to v
    //! Does nothing if v isn't actually a decrease.
    inline bool check_decrease_value(int k, const P &v) { return check_decrease_value(super::node(k), v); }

  protected:
    //! protected variant of `check_decrease_value` using a node pointer
    inline bool check_decrease_value(N *n, const P &v) {
        minorminer_assert(n != nullptr);
        if (super::current(n) && n->active()) {
            if (v < n->val) {
                decrease_value(n, v);
                return true;
            } else {
                return false;
            }
        } else {
            insert_value(n, v);
            return true;
        }
    }

  public:
    //! set the value associated with `k` to `v`
    inline void set_value(int k, const P &v) { set_value(super::node(k), v); }

  protected:
    //! protected variant of `set_value` using a node pointer
    inline void set_value(N *n, const P &v) {
        minorminer_assert(n != nullptr);
        if (super::current(n) && n->active()) {
            if (v < n->val) {
                decrease_value(n, v);
            } else if (n->val < v) {
                increase_value(n, v);
            }
        } else {
            insert_value(n, v);
        }
    }

  public:
    //! Decrease the value of k to v
    //! NOTE: Assumes that v is lower than the current value of k
    inline void decrease_value(int k, const P &v) { decrease_value(super::node(k), v); }

  protected:
    //! protected variant of `decrease_value` using a node pointer
    inline void decrease_value(N *n, const P &v) {
        minorminer_assert(n != nullptr);
        minorminer_assert(v < n->val);

        n->val = v;
        restructure_decrease(n);
    }

    //! protected variant of `decrease_value` using a node pointer
    inline void increase_value(N *n, const P &v) {
        minorminer_assert(n != nullptr);
        minorminer_assert(v > n->val);

        n->val = v;
        restructure_increase(n);
    }

    inline void insert_value(N *n, const P &v) {
        minorminer_assert(n != nullptr);

        n->val = v;
        super::root = n->merge_roots(super::root);
    }

    //! update the data structure to reflect a decrease in the value of `a`
    inline void restructure_decrease(N *a) {
        minorminer_assert(a != nullptr);
        if (a != super::root) {
            a->extract_root();
            super::root = super::root->merge_roots(a);
        } else {
            minorminer_assert(a->prev == nullptr);
        }
    }

    //! update the data structure to reflect a increase in the value of `a`
    inline void restructure_increase(N *a) {
        minorminer_assert(a != nullptr);
        if (a != super::root) {
            a->extract_root();
            a = a->increase_root();
            super::root = super::root->merge_roots(a);
        } else {
            super::root = super::root->increase_root();
        }
    }
};

#undef nullval
#undef max_P
}
