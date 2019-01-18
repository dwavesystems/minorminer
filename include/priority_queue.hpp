#pragma once
namespace find_embedding {

class min_heap_tag {};
class max_heap_tag {};

template <typename P, typename heap_tag>
class priority_node {
  public:
    int node;
    int dirt;
    P dist;

    priority_node(int n, int r, P d) : node(n), dirt(r), dist(d) {}
    bool operator<(const priority_node<P, heap_tag> &b) const {
        if (std::is_same<min_heap_tag, heap_tag>::value)
            return (dist > b.dist) || ((dist == b.dist) && (dirt > b.dirt));
        if (std::is_same<max_heap_tag, heap_tag>::value)
            return (dist < b.dist) || ((dist == b.dist) && (dirt < b.dirt));
    }
};
}
