#pragma once
namespace find_embedding {

template <typename P>
class dirty_priority_node {
  public:
    int node;
    int dirt;
    P dist;

    dirty_priority_node(int n, int r, P d) : node(n), dirt(r), dist(d) {}

    bool operator<(const dirty_priority_node<P> &b) const {
        return (dist > b.dist) || ((dist == b.dist) && (dirt > b.dirt));
    }
};

template <typename P>
class priority_node {
  public:
    int node;
    P dist;

    priority_node(int n, P d) : node(n), dist(d) {}

    bool operator<(const priority_node<P> &b) const { return (dist > b.dist) || ((dist == b.dist) && (node > b.node)); }
};
}
