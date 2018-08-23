#pragma once
#include "stl_clib.h"

#ifdef __cplusplus
#include <limits>
extern "C" {
#endif
  typedef struct optional_parameters_clib {
    int max_no_improvement;
    int timeout;
    int tries;
    int verbose;
    int inner_rounds;
    int max_fill;
    int return_overlap;
    int chainlength_patience;
    int threads;
    int skip_initialization;
    mivi* fixed_chains;
    mivi* initial_chains;
    mivi* restrict_chains;
    int (*cancelled_callback) ();
    void (*display_callback) (const char*);
  } optional_parameters_clib;
  optional_parameters_clib* make_default_optional_parameters( mivi*, mivi*, mivi*);
  void delete_optional_parameters(optional_parameters_clib* o);
#ifdef __cplusplus
}
#endif
