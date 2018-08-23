#include "optional_parameters_clib.h"
#include "local_interaction_clib.h"

optional_parameters_clib* make_default_optional_parameters(mivi* fixed_chains, mivi* restrict_chains, mivi* initial_chains) {
  optional_parameters_clib* o = (optional_parameters_clib*)malloc(sizeof(optional_parameters_clib));
  o->timeout = 1000.0;
  o->verbose = 0;
  o->max_no_improvement = 10;
  o->tries = 10;
  o->inner_rounds = std::numeric_limits<int>::max();
  o->max_fill = std::numeric_limits<int>::max();
  o->return_overlap = false;
  o->chainlength_patience = 2;
  o->threads = 1;
  o->skip_initialization = false;
  o->fixed_chains = fixed_chains==NULL?make_mivi():fixed_chains;
  o->initial_chains = initial_chains==NULL?make_mivi():initial_chains;
  o->restrict_chains = restrict_chains==NULL?make_mivi():restrict_chains;
  o->display_callback = &displayOutput_clib;
  o->cancelled_callback = &isCancelled;
  return o;
}

void delete_optional_parameters(optional_parameters_clib* o) {
  free(o);
}
