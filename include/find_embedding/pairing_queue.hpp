// Copyright 2019 - 2020 D-Wave Systems Inc.
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
#include <stdexcept>
namespace find_embedding {

class min_heap_tag {};
class max_heap_tag {};

template <typename P, typename heap_tag = min_heap_tag>
class priority_node {
  public:
    int node;
    int dirt;
    P dist;
    priority_node() {}
    priority_node(int n, int r, P d) : node(n), dirt(r), dist(d) {}
    bool operator<(const priority_node<P, heap_tag> &b) const {
        if (std::is_same<min_heap_tag, heap_tag>::value)
            return (dist > b.dist) || ((dist == b.dist) && (dirt > b.dirt));
        if (std::is_same<max_heap_tag, heap_tag>::value)
            return (dist < b.dist) || ((dist == b.dist) && (dirt < b.dirt));
    }
};

template <typename N>
class pairing_node : public N {
    pairing_node *next;
    pairing_node *desc;

  public:
    pairing_node<N>() {}

    template <class... Args>
    pairing_node<N>(Args... args) : N(args...), next(nullptr), desc(nullptr) {}

    //! the basic operation of the pairing queue -- put `this` and `other`
    //! into heap-order
    inline pairing_node<N> *merge_roots(pairing_node<N> *other) {
        if (other == nullptr) return this;

        other = merge_roots_unsafe(other);

        other->next = nullptr;
        return other;
    }

    template <class... Args>
    void refresh(Args... args) {
        this->~pairing_node<N>();
        new (this) pairing_node<N>(args...);
    }

    inline pairing_node<N> *next_root() { return desc; }

  private:
    //! the basic operation of the pairing queue -- put `this` and `other`
    //! into heap-order
    inline pairing_node<N> *merge_roots_unsafe(pairing_node<N> *other) {
        if (*other < *this)
            return merge_roots_unchecked(other);
        else
            return other->merge_roots_unchecked(this);
    }

    //! merge_roots, assuming `other` is not null and that `val` < `other->val`.
    //!  may invalidate the internal data structure (see source for details)
    inline pairing_node<N> *merge_roots_unchecked(pairing_node *other) {
        // this very unsafe version of self.merge_roots which
        // * doesn't check for nullval
        // * doesn't ensure that the returned node has next = nullval
        // * doesn't check that this < other
        minorminer_assert(other != nullptr);
        minorminer_assert(*other < *this);

        other->next = desc;
        desc = other;
        return this;
    }

  public:
    inline pairing_node<N> *merge_pairs() {
        pairing_node<N> *a = this;
        pairing_node<N> *r = next;
        if (r == nullptr) {
            return a;
        } else {
            pairing_node<N> *c = r->next;
            r->next = nullptr;
            r = a->merge_roots_unsafe(r);
            r->next = nullptr;
            a = c;
        }
        while (a != nullptr) {
            pairing_node<N> *b = a->next;
            if (b == nullptr) {
                return a->merge_roots_unsafe(r);
            } else {
                pairing_node<N> *c = b->next;
                b = a->merge_roots_unsafe(b);
                b->next = nullptr;
                r = b->merge_roots_unsafe(r);
                a = c;
            }
        }
        return r;
    }
};

template <typename N>
class pairing_queue {
    int count;
    int size;
    pairing_node<N> *root;
    pairing_node<N> *mem;

  public:
    pairing_queue(int n) : count(0), size(n), root(nullptr), mem(new pairing_node<N>[n]) {}

    pairing_queue(pairing_queue &&other) : count(other.count), size(other.size), root(other.root), mem(other.mem) {
        other.mem = nullptr;
    }

    ~pairing_queue() {
        if (mem != nullptr) delete[] mem;
    }

    inline void reset() {
        root = nullptr;
        count = 0;
    }

    inline bool empty() { return root == nullptr; }

    template <class... Args>
    inline void emplace(Args... args) {
        pairing_node<N> *x = mem + (count++);
        x->refresh(args...);
        root = x->merge_roots(root);
    }

    inline N top() { return static_cast<N>(*root); }

    inline void pop() {
        root = root->next_root();
        if (root == nullptr) return;
        root = root->merge_pairs();
    }
};
}
