#ifdef __cplusplus
  #include "../include/graph.hpp"
  using namespace graph;
  extern "C" {
#else
  typedef struct input_graph input_graph;
#endif
  input_graph* make_graph();
  void delete_graph(input_graph*);
  void push_back(input_graph*,int,int);
  void clear(input_graph*);
  int num_nodes(input_graph*);
  int num_edges(input_graph*);
#ifdef __cplusplus
}
#endif 
