# Copyright 2020 D-Wave Systems Inc.
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

import time

import networkx as nx
import minorminer as mm

from .layout import Layout, dnx_layout, p_norm
from .placement import Placement, closest, intersection


def find_embedding(
    S,
    T,
    layout=(p_norm, None),
    placement=closest,
    mm_hint_type="initial_chains",
    return_layouts=False,
    **kwargs
):
    """Tries to embed S in T by computing layout-aware chains and passing 
    them to :func:`minorminer.find_embedding`. Chains are passed as either 
    `initial_chains` or `suspend_chains` (see documentation for 
    :func:`minorminer.find_embedding` to learn more).

    Args:
        S (NetworkX Graph/edges data structure (dict, list, ...)):
            The source graph being embedded or a NetworkX supported data structure
            for edges (see :func:`nx.convert.to_networkx_graph` for details).

        T (NetworkX Graph/edges data structure (dict, list, ...)):
            The target graph being embedded into or a NetworkX supported data 
            structure for edges (see :func:`nx.convert.to_networkx_graph` for 
            details).

        layout (function/(function/dict/Layout, function/dict/Layout), optional):
            A function to compute the :class:`.Layout` for both S and T, or a 
            2-tuple that either consists of a pair of functions or pre-computed 
            layouts (in the form of :class:`.Layout` or dicts). The first entry 
            in the 2-tuple applies to S while the second applies to T.

            Note:
                If ``layout`` is a single function and T is a dnx_graph, then the 
                function passed in is only applied to S and the dnx_layout is 
                applied to T. To run a layout function explicitly on T, pass 
                it in as a 2-tuple; i.e. (p_norm, p_norm).

        placement (function/dict, optional, default=minorminer.placement.closest):
            A function that uses the layouts of S and T to map the vertices 
            of S to subsets of vertices of T (:class:`.Placement`), or a dict 
            that contains the precomputed mapping/:class:`.Placement`.
            
            By default, :func:`~placement.closest` is called to compute placement.

        mm_hint_type (str, optional, default="initial_chains"):
            This is the hint type passed to :func:`minorminer.find_embedding`. 
            Supported types are "initial_chains" and "suspend_chains". See
            :func:`minorminer.find_embedding` for more information.
        
        return_layouts (bool, optional, default=False):
            If True, layout objects of S and T are also returned.

        **kwargs (dict):
            Keyword arguments passed to :class:`.Layout`, :class:`.Placement` 
            or :func:`minorminer.find_embedding`.

    Returns:
        dict: An embedding of vertices of S (keys) to chains in T (values). This 
        embedding is dependent on the kwargs being passed in. If ``return_layouts``
        is True, a 2-tuple is returned in which the first element is the embedding 
        dict and the second element is another 2-tuple containing the source and 
        target :class:`.Layout` objects.

    """
    start = time.perf_counter()

    # Parse kwargs
    layout_kwargs, placement_kwargs = _parse_kwargs(kwargs)

    # Compute the layouts
    S_layout, T_layout = _parse_layout_parameter(S, T, layout, layout_kwargs)

    # Compute the placement (i.e. chains)
    S_T_placement = Placement(
        S_layout, T_layout, placement, **placement_kwargs)

    end = time.perf_counter()
    timeout = kwargs.get("timeout")
    if timeout is not None:
        time_remaining = timeout - (end - start)
        if time_remaining <= 0:
            raise TimeoutError(
                "Layout and placement took {}s, you set timeout at {}.".format(end-start, timeout))
        kwargs["timeout"] = time_remaining

    # Run minorminer.find_embedding
    if mm_hint_type == "initial_chains":
        output = mm.find_embedding(
            S, T, initial_chains=S_T_placement, **kwargs)
    elif mm_hint_type == "suspend_chains":
        output = mm.find_embedding(S, T, suspend_chains={
            v: [C] for v, C in S_T_placement.items()}, **kwargs)
    else:
        raise ValueError(
            "Only initial_chains and suspend_chains are supported minorminer hint types.")

    if return_layouts:
        return output, (S_layout, T_layout)
    return output


def _parse_kwargs(kwargs):
    """Extract kwargs for layout and placement functions. Leave the remaining 
    ones for :func:`minorminer.find_embedding`.
    """
    layout_kwargs = {}
    # For the layout object
    if "dim" in kwargs:
        layout_kwargs["dim"] = kwargs.pop("dim")
    if "center" in kwargs:
        layout_kwargs["center"] = kwargs.pop("center")
    if "scale" in kwargs:
        layout_kwargs["scale"] = kwargs.pop("scale")

    placement_kwargs = {}
    # For the placement object
    if "scale_ratio" in kwargs:
        placement_kwargs["scale_ratio"] = kwargs.pop("scale_ratio")
    # For closest strategy
    if "subset_size" in kwargs:
        placement_kwargs["subset_size"] = kwargs.pop("subset_size")
    if "num_neighbors" in kwargs:
        placement_kwargs["num_neighbors"] = kwargs.pop("num_neighbors")

    return layout_kwargs, placement_kwargs


def _parse_layout_parameter(S, T, layout, layout_kwargs):
    """Determine what combination of iterable, dict, and function the layout 
    parameter is.
    """
    if nx.utils.iterable(layout):
        try:
            s_layout, t_layout = layout
        except ValueError:
            raise ValueError(
                "layout is expected to be a function or a length-2 iterable")
    else:
        s_layout = t_layout = layout

    # Get a Layout object for S
    if isinstance(s_layout, Layout):
        S_layout = s_layout
    else:
        # Assumes s_layout a callable or implements a mapping interface
        S_layout = Layout(S, layout=s_layout, **layout_kwargs)

    # Get the Layout object for T
    if isinstance(t_layout, Layout):
        T_layout = t_layout
    else:
        # Use the dnx_layout if possible
        if T.graph.get("family") in ("chimera", "pegasus", "zephyr"):
            T_layout = Layout(T, layout=dnx_layout, **layout_kwargs)
        # Assumes t_layout a callable or implements a mapping interface
        else:
            T_layout = Layout(T, layout=t_layout, **layout_kwargs)

    return S_layout, T_layout
