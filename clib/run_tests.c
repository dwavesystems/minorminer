/* #include "stl_clib.h" */
/* #include "graph_clib.h" */
/* #include "local_interaction_clib.h" */
/* #include "find_embedding_clib.h" */
#include "minorminer_clib.h"
#include <stdio.h>

int failed = 0;

#define ASSERT(a) if(!(a)) { printf("FAILED: %s\n", #a); failed++; }
#define RUNTEST(test) { int f = failed; test; if(f!=failed){ printf("FAILED TEST: %s\n", #test); } }

int gotcallback = 0;
void test_display_callback (const char *str) {
  /*printf("%s\n",str);*/ /* Useful for debugging, see the output of the embedder */
  gotcallback = 1;
}

void test_embedding () {
  input_graph* g;
  input_graph* h;
  vvi *result;
  optional_parameters_clib *o = make_default_optional_parameters(NULL,NULL,NULL);
  int success;
  ASSERT(g = make_graph());
  ASSERT(h = make_graph());
  ASSERT(result = make_vvi());
  /* Embed a triangle into a square! */
  push_back(g,0,1);
  push_back(g,1,2);
  push_back(g,2,0);
  push_back(h,0,1);
  push_back(h,1,2);
  push_back(h,2,3);
  push_back(h,3,0);
  o->verbose = 3;
  o->display_callback = &test_display_callback;
  success = findembedding_clib(g, h, o, result);
  ASSERT(success==1);
  ASSERT(vvi_length(result) == 3);
  /* Check to make sure one chain is length 2 */
  ASSERT(vectorint_length(vvi_at(result,0)) == 2 || vectorint_length(vvi_at(result,1)) == 2 || vectorint_length(vvi_at(result,2)) == 2);
  ASSERT((vectorint_length(vvi_at(result,0)) +  vectorint_length(vvi_at(result,1)) + vectorint_length(vvi_at(result,2))) == 4);
  /* vvi_print(result);*/
}

void test_cancellation () {
  ASSERT(isCancelled() == 0);
  setCancelled();
  ASSERT(isCancelled() == 1);
  resetCancelled();
  ASSERT(isCancelled() == 0);
}

void graph_test () {
  input_graph* g;
  ASSERT(g = make_graph());
  ASSERT(num_edges(g) == 0);
  ASSERT(num_nodes(g) == 0);
  push_back(g,3,4);
  ASSERT(num_edges(g) == 1);
  ASSERT(num_nodes(g) == 5);
  delete_graph(g);
}

void stl_test() {
  vectorint* v;
  vvi* vv;
  int *i;
  ASSERT(v = make_vectorint());
  ASSERT(vv = make_vvi());
  ASSERT(vectorint_length(v) == 0);
  ASSERT(vvi_length(vv) == 0);
  vectorint_push_back(v,1);
  vectorint_push_back(v,2);
  vectorint_push_back(v,3);
  ASSERT(vectorint_length(v) == 3);
  i = vectorint_to_arrayint(v);
  ASSERT(i[0] == 1);
  ASSERT(i[1] == 2);
  ASSERT(i[2] == 3);
  vvi_push_back(vv,v);
  ASSERT(vvi_length(vv) == 1);
  ASSERT(vectorint_to_arrayint(vvi_at(vv,0))[0] == 1);
}

int mivi_callback_called = 0;

void mivi_callback_test(int i, vectorint* vi) {
  mivi_callback_called += 1;
  ASSERT(vectorint_length(vi)==1)
  ASSERT(vectorint_to_arrayint(vi)[0]==123)
}

void test_mivi () {
  vectorint* a;
  vectorint* b;
  mivi* m;
  ASSERT(a = make_vectorint())
  ASSERT(b = make_vectorint())
  ASSERT(m = make_mivi())
  vectorint_push_back(a,123);
  vectorint_push_back(b,123);
  ASSERT(vectorint_length(a)==1);
  ASSERT(vectorint_length(b)==1);
  mivi_set(m,1,a);
  mivi_set(m,2,b);
  mivi_map(m, &mivi_callback_test);
  ASSERT(mivi_callback_called == 2);
  delete_vectorint(a);
  delete_vectorint(b);
  delete_mivi(m);
}

int main(int argc, char**argv) {
  printf("Running tests\n");
  RUNTEST(graph_test());
  RUNTEST(test_cancellation());
  RUNTEST(stl_test());
  RUNTEST(test_embedding());
  RUNTEST(test_mivi());
  if(failed > 0) {
    printf("Some tests failed\n");
    return -1;
  } else {
    printf("All tests passed\n");
    return 0;
  }
}
