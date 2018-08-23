#include "find_embedding_clib.h"
#include "local_interaction_clib.h"

int findembedding_clib(input_graph *var_g_, input_graph *qubit_g_, optional_parameters_clib *params_,
                       vvi *chains_) {
  /*Find an embedding of a smaller graph, var_g, into a larger graph qubit_g.  Resulting chains are placed in the vector chains.*/
  graph::input_graph &var_g = *var_g_;
  graph::input_graph &qubit_g = *qubit_g_;
  find_embedding::optional_parameters *params = new find_embedding::optional_parameters();
  clib_LocalInteraction* c = new clib_LocalInteraction();
  c->display_callback = params_->display_callback;
  c->cancelled_callback = params_->cancelled_callback;
  params->localInteractionPtr.reset(c);
  params->timeout = params_->timeout;
  params->verbose = params_->verbose;
  params->max_no_improvement = params_->max_no_improvement;
  params->tries = params_->tries;
  params->inner_rounds = params_->inner_rounds;
  params->max_fill = params_->max_fill;
  params->threads = params_->threads;
  params->chainlength_patience = params_->chainlength_patience;
  params->return_overlap = params_->return_overlap;
  params->skip_initialization = params_->skip_initialization;
  params->restrict_chains = *params_->restrict_chains;
  params->fixed_chains = *params_->fixed_chains;
  params->initial_chains = *params_->initial_chains; 
  return findEmbedding(var_g,qubit_g,*params,*chains_);
}
