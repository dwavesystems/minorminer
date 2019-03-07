#include "graph.hpp"
#include "gtest/gtest.h"
#include "util.hpp"

// Construct a graph, clear the graph
TEST(input_graph, construction_empty) {
    // Build an empty graph
    graph::input_graph graph;
    EXPECT_EQ(graph.num_nodes(), 0);
    EXPECT_EQ(graph.num_edges(), 0);

    // Expand the graph
    graph.push_back(0, 1);
    graph.push_back(1, 0);
    EXPECT_EQ(graph.num_nodes(), 2);
    EXPECT_EQ(graph.num_edges(), 2);

    // Clear the graph
    graph.clear();
    EXPECT_EQ(graph.num_nodes(), 0);
    EXPECT_EQ(graph.num_edges(), 0);
}

TEST(input_graph, construction_small) {
    // Build a simple graph
    graph::input_graph graph(4, {0, 0, 0, 1}, {1, 2, 3, 2});
    EXPECT_EQ(graph.num_nodes(), 4);
    EXPECT_EQ(graph.num_edges(), 4);

    // Expand the graph
    graph.push_back(2, 4);
    EXPECT_EQ(graph.num_nodes(), 5);
    EXPECT_EQ(graph.num_edges(), 5);

    // Clear the graph
    graph.clear();
    EXPECT_EQ(graph.num_nodes(), 0);
    EXPECT_EQ(graph.num_edges(), 0);
}

// Get neighbors of a simple graph for null case, single case, multiple case
TEST(input_graph, neighbors_empty) {
    // Build an empty graph
    graph::input_graph graph;
    std::vector<std::vector<int>> neighbors = graph.get_neighbors();
    EXPECT_EQ(neighbors.size(), 0);

    neighbors = graph.get_neighbors(std::vector<int>({}));
    EXPECT_EQ(neighbors.size(), 0);
}

TEST(input_graph, neighbors_node) {
    // Build a simple graph
    graph::input_graph graph(1, {0}, {0});

    std::vector<std::vector<int>> neighbors = graph.get_neighbors();
    EXPECT_EQ(neighbors.size(), 1);
    EXPECT_EQ(neighbors[0].size(), 0);

    neighbors = graph.get_neighbors(std::vector<int>({0}));
    EXPECT_EQ(neighbors.size(), 1);
    EXPECT_EQ(neighbors[0].size(), 0);
}

TEST(input_graph, neighbors_small) {
    // Build a simple graph
    graph::input_graph graph(4, {0, 0, 0, 1}, {1, 2, 3, 2});

    std::vector<std::vector<int>> neighbors = graph.get_neighbors();
    EXPECT_EQ(neighbors.size(), 4);
    EXPECT_EQ(neighbors[0].size(), 3);
    EXPECT_EQ(neighbors[0], std::vector<int>({1, 2, 3}));
    EXPECT_EQ(neighbors[1].size(), 2);
    EXPECT_EQ(neighbors[1], std::vector<int>({0, 2}));
    EXPECT_EQ(neighbors[2].size(), 2);
    EXPECT_EQ(neighbors[2], std::vector<int>({0, 1}));
    EXPECT_EQ(neighbors[3].size(), 1);
    EXPECT_EQ(neighbors[3], std::vector<int>({0}));
}

TEST(input_graph, neighbors_small_relabel) {
    // Build a simple graph
    graph::input_graph graph(4, {0, 0, 0, 1}, {1, 2, 3, 2});

    std::vector<std::vector<int>> neighbors = graph.get_neighbors(std::vector<int>({3, 2, 0, 1}));
    EXPECT_EQ(neighbors.size(), 4);
    EXPECT_EQ(neighbors[3].size(), 3);
    EXPECT_EQ(neighbors[3], std::vector<int>({0, 1, 2}));
    EXPECT_EQ(neighbors[2].size(), 2);
    EXPECT_EQ(neighbors[2], std::vector<int>({0, 3}));
    EXPECT_EQ(neighbors[0].size(), 2);
    EXPECT_EQ(neighbors[0], std::vector<int>({2, 3}));
    EXPECT_EQ(neighbors[1].size(), 1);
    EXPECT_EQ(neighbors[1], std::vector<int>({3}));
}

//
// // produce the node->nodelist mapping for our graph, where certain nodes are
// // marked as sources (no incoming edges)
// void get_neighbors_sources(vector<vector<int> > &nbrs, vector<int> &sources) {
//     vector<set<int>> _nbrs(_num_nodes);
//
// // produce the node->nodelist mapping for our graph, where certain nodes are
// // marked as sinks (no outgoing edges)
// void get_neighbors_sinks(vector<vector<int> > &nbrs, vector<int> &sinks) {
//
//
// // produce the node->nodelist mapping for our graph, where certain nodes are
// // marked as sinks (no outgoing edges), relabeling all nodes along the way
// void get_neighbors_sinks_relabel(vector<vector<int> > &nbrs, vector<int> &sinks, vector<int> &relabel) {
