# Copyright 2025 D-Wave
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
#
# ================================================================================================

"""
Tools for embedding algorithms of lattices on Zephyr-topology graphs.

This package includes:

1. Auxiliary coordinate types to work with Zephyr nodes, external paths, and quotient external paths.
2. A chain object representing a pair of Zephyr nodes—one vertical and one horizontal—
   typically used to form chain-length-2 embeddings of lattices on Zephyr.
3. A graph object whose nodes are the chains described above, enabling
   connectivity checks based on z-coupling.
4. A lattice survey module that provides utilities for embedding lattices
   on partially-yielded Zephyr graphs.
5. A qoutient tile object to facilitate embedding lattice-type graphs on Zephyr.
"""

from minorminer._lattice_utils.auxiliary_coordinates import *
from minorminer._lattice_utils.survey import *
from minorminer._lattice_utils.qtile import *
