#include "graph_clib.h"

input_graph* make_graph() {
  return new input_graph();
}

void delete_graph(input_graph* g) {
  delete(g);
}

int num_edges(input_graph* g) {
  return g->num_edges();
}

int num_nodes(input_graph* g) {
  return g->num_nodes();
}

void push_back(input_graph* g, int i, int j) {
  return g->push_back(i,j);
}

