#pragma once
#include "stl_clib.h"
#include "graph_clib.h"
#include "optional_parameters_clib.h"
#ifdef __cplusplus
#include "../include/find_embedding.hpp"
extern "C" {
#endif
  int findembedding_clib(input_graph *, input_graph *, optional_parameters_clib *, vvi*);
#ifdef __cplusplus
}
#endif

