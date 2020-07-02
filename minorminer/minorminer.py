# Copyright 2019 - 2020 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import absolute_import as __absolute_import
from minorminer._minorminer import miner, VARORDER, find_embedding as __find_embedding
from functools import wraps as __wraps

# This wrapper exists to overcome a curious limitation of Cython, and make
# find_embedding friendlier for the inspect module.
@__wraps(__find_embedding)
def find_embedding(S, T,
                   max_no_improvement=10,
                   random_seed=None,
                   timeout=1000,
                   max_beta=None,
                   tries=10,
                   inner_rounds=None,
                   chainlength_patience=10,
                   max_fill=None,
                   threads=1,
                   return_overlap=False,
                   skip_initialization=False,
                   verbose=0,
                   interactive=False,
                   initial_chains=(),
                   fixed_chains=(),
                   restrict_chains=(),
                   suspend_chains=(),
                   ):
    return __find_embedding(S, T,
                            max_no_improvement=max_no_improvement,
                            random_seed=random_seed,
                            timeout=timeout,
                            max_beta=max_beta,
                            tries=tries,
                            inner_rounds=inner_rounds,
                            chainlength_patience=chainlength_patience,
                            max_fill=max_fill,
                            threads=threads,
                            return_overlap=return_overlap,
                            skip_initialization=skip_initialization,
                            verbose=verbose,
                            interactive=interactive,
                            initial_chains=initial_chains,
                            fixed_chains=fixed_chains,
                            restrict_chains=restrict_chains,
                            suspend_chains=suspend_chains,
                            )
