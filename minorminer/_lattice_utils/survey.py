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


from collections import namedtuple
from functools import cached_property
from typing import Callable, Literal

import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler
from minorminer.utils.zephyr.node_edge import Edge

from minorminer._lattice_utils.auxiliary_coordinates import UWJ, UWKJ, UWKJZ

__all__ = ["ZLatticeSurvey", "ZSE"]

ZSE = namedtuple("ZSE", ["z_start", "z_end"])


class ZLatticeSurvey:
    """
    A class that provides convenient representations and helpers
    for embedding algorithms of lattices on Zephyr topology.
    Enables calculating the stretch of external paths.
    Takes a Zephyr graph or DWaveSampler with Zephyr topology.

        Args:
            G (nx.Graph | DWaveSampler): A graph or DWaveSampler with Zephyr topology.

    Example:
    >>> from dwave_networkx import zephyr_graph
    >>> from minorminer.cube_embedding._lattice_embedding.lattice_survey import ZLatticeSurvey
    >>> m = 3
    >>> G = zephyr_graph(m=m, t=4)
    >>> G.remove_node(1)
    >>> G.remove_node(20)
    >>> lsurvey = ZLatticeSurvey(G)
    >>> print(f"The number of missing nodes is {lsurvey.num_missing_nodes}. The missing nodes are {lsurvey.missing_nodes}")
    The number of missing nodes is 2. The missing nodes are {UWKJZ(u=0, w=0, k=3, j=0, z=2), UWKJZ(u=0, w=0, k=0, j=0, z=1)}
    >>> for uwj, uwj_dict in lsurvey_external_paths_stretch.items():
    ...     for k, z_stretches in uwj_dict.items():
    ...         if z_stretches != [ZSE(z_start=0, z_end=m - 1)]:
    ...             external_path = UWKJ(u=uwj.u, w=uwj.w, k=k, j=uwj.j)
    ...             print(f"The external path {external_path} does not extend across the full z-range.")
    The external path UWKJ(u=0, w=0, k=0, j=0) does not extend across the full z-range.
    The external path UWKJ(u=0, w=0, k=3, j=0) does not extend across the full z-range.
    """

    def __init__(
        self,
        G: nx.Graph | DWaveSampler,
    ) -> None:

        self._m, self._t, self._label = self.get_shape_coord(G)
        if isinstance(G, nx.Graph):
            G_nodes = list(G.nodes())
            G_edges = list(G.edges())
        elif isinstance(G, DWaveSampler):
            G_nodes = G.nodelist
            G_edges = G.edgelist
        self._nodes: set[UWKJZ] = {UWKJZ(*self._label_to_coord(v)) for v in G_nodes}
        self._edges: set[Edge] = {
            Edge(UWKJZ(*self._label_to_coord(u)), UWKJZ(*self._label_to_coord(v)))
            for (u, v) in G_edges
        }
        self._external_paths_stretch = None

    @staticmethod
    def get_shape_coord(
        G: nx.Graph | DWaveSampler,
    ) -> tuple[int, int, Literal["int", "coordinate"]]:
        """
        Returns the shape and coordinates of G, which must be a zephyr graph or
        DWaveSampler with zephyr topology.

        Args:
            G (nx.Graph | DWaveSampler): A zephyr graph or DWaveSampler with zephyr topology.

        Returns:
            tuple[int, int, Literal["int", "coordinate"]]:
                - 0-th index indicates the grid size of G (m).
                - 1-st index indicates the tile size of G (t).
                - 2-nd index is 'int' if the node lables of ``G`` are integers;
                    it is 'coordinate' if the node lables of ``G`` are Zephyr coordinates.
        """
        if isinstance(G, nx.Graph):
            G_info = G.graph
            G_top = G_info.get("family")
            if G_top != "zephyr":
                raise ValueError(f"Expected a graph with zephyr topology, got {G_top}")
            return G_info.get("rows"), G_info.get("tile"), G_info.get("labels")

        elif isinstance(G, DWaveSampler):
            sampler_top: dict[str, str | int] = G.properties.get("topology")
            if sampler_top.get("type") != "zephyr":
                raise ValueError(f"Expected a sampler with zephyr topology, got {sampler_top}")
            return (*sampler_top.get("shape"), "int")

    @cached_property
    def _label_to_coord(self) -> Callable[[int | tuple[int]], tuple[int]]:
        """Returns a function that converts the linear or zephyr coordinates to
        the corresponding zephyr coordinates"""
        if self._label == "int":
            return dnx.zephyr_coordinates(m=self._m, t=self._t).linear_to_zephyr
        elif self._label == "coordinate":
            return lambda v: v

    @property
    def m(self) -> int:
        """Returns the grid size of ``G``."""
        return self._m

    @property
    def t(self) -> int:
        """Returns the tile size of ``G``."""
        return self._t
    
    @property
    def label(self) -> str:
        return self._label

    @property
    def nodes(self) -> set[UWKJZ]:
        """Returns the ``ZNode``s of the sampler/graph."""
        return self._nodes

    @cached_property
    def missing_nodes(self) -> set[UWKJZ]:
        """Returns the ZNodes of the sampler/graph which are missing compared to perfect yield
        Zephyr graph on the same shape.
        """
        parent_nodes = [
            UWKJZ(*v) for v in dnx.zephyr_graph(m=self._m, t=self._t, coordinates=True).nodes()
        ]
        return {v for v in parent_nodes if not v in self._nodes}

    @property
    def edges(self) -> set[Edge]:
        """Returns the ZEdges of the sampler/graph"""
        return self._edges

    @cached_property
    def missing_edges(self) -> set[Edge]:
        """Returns the ZEdges of the sampler/graph which are missing compared to
        perfect yield Zephyr graph on the same shape."""
        parent_edges = [
            Edge(UWKJZ(*u), UWKJZ(*v))
            for (u, v) in dnx.zephyr_graph(m=self._m, t=self._t, coordinates=True).edges()
        ]
        return {e for e in parent_edges if not e in self._edges}

    @property
    def extra_missing_edges(self) -> set[Edge]:
        """Returns the Edges of the sampler/graph which are missing compared to
        perfect yield Zephyr graph on the same shape and are not incident with
        a missing node."""
        return {e for e in self.missing_edges if e[0] in self._nodes and e[1] in self._nodes}

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes"""
        return len(self._nodes)

    @property
    def num_missing_nodes(self) -> int:
        """Returns the number of missing nodes"""
        return len(self.missing_nodes)

    @property
    def num_edges(self) -> int:
        """Returns the number of edges"""
        return len(self._edges)

    @property
    def num_missing_edges(self) -> int:
        """Returns the number of missing edges"""
        return len(self.missing_edges)

    @property
    def num_extra_missing_edges(self) -> int:
        """Returns the number of missing edges that are not incident
        with a missing node"""
        return len(self.extra_missing_edges)

    def _ext_path(self, uwkj: UWKJ) -> list[ZSE]:
        """Returns uwkj_sur, where uwkj_sur contains ZSE(z_start, z_end)
        for each non-overlapping external path segment of uwkj.
        """
        z_vals = list(range(self._m))  # As in zephyr coordinates

        def _ext_seg(z_start: int) -> ZSE | None:
            """If (u, w, k, j, z_start) does not exist, returns None.
            Else, finds the external path segment starting at z_start going to right, and
            returns the endpoints of the segment (z_start, z_end).
            """
            cur = UWKJZ(*uwkj, z_start)
            if cur in self.missing_nodes:
                return None

            upper_z: int = z_vals[-1]
            if z_start > upper_z:
                return None

            if z_start == upper_z:
                return ZSE(z_start, z_start)

            next_ext = UWKJZ(*uwkj, z_start + 1)
            ext_edge = Edge(cur, next_ext)
            if ext_edge in self.missing_edges:
                return ZSE(z_start, z_start)

            is_extensible = _ext_seg(z_start + 1)
            if is_extensible is None:
                return ZSE(z_start, z_start)

            return ZSE(z_start, is_extensible.z_end)

        uwkj_sur: list[ZSE] = []
        while z_vals:
            z_start = z_vals[0]
            seg = _ext_seg(z_start=z_start)
            if seg is None:
                z_vals.remove(z_start)
            else:
                uwkj_sur.append(seg)
                for i in range(seg.z_start, seg.z_end + 1):
                    z_vals.remove(i)

        return uwkj_sur

    def calculate_external_paths_stretch(self) -> dict[UWKJ, list[ZSE]]:
        """Calculates the stretch of external paths of the graph/sampler.

        Returns:
        dict[UWJ, dict[int, list[ZSE]]]:
            A nested dictionary of the form:
                { uwj: { k: [ZSE(z_start, z_end), ...] } }

            - Outer keys are quotients of external paths, represented as ``UWJ``.
            - Each ``UWJ`` maps to a dictionary whose:
                - Keys are ``k`` values from external paths (``UWKJ``) such that ``UWKJ.uwj == uwj``.
                - Values are lists of ``ZSE`` objects representing the maximal connected z-segments
                within the corresponding external path.
        """
        if self._external_paths_stretch is None:
            uwj_vals = [
                UWJ(u=u, w=w, j=j)
                for u in range(2)  # As in zephyr coordinates
                for w in range(2 * self._m + 1)  # As in zephyr coordinates
                for j in range(2)  # As in zephyr coordinates
            ]
            k_vals = list(range(self._t))  # As in zephyr coordinates
            self._external_paths_stretch = {
                uwj: {
                    k_idx: self._ext_path(UWKJ(u=uwj.u, w=uwj.w, k=k_idx, j=uwj.j))
                    for k_idx in k_vals
                }
                for uwj in uwj_vals
            }
        return self._external_paths_stretch
