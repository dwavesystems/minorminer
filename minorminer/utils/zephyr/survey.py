# Copyright 2025 D-Wave Systems Inc.
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


from __future__ import annotations

from collections import namedtuple
from functools import cached_property
from typing import Callable

import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler
from minorminer.utils.zephyr.coordinate_systems import ZephyrCoord
from minorminer.utils.zephyr.node_edge import ZEdge, ZNode, ZShape

UWKJ = namedtuple("UWKJ", ["u", "w", "k", "j"])
ZSE = namedtuple("ZSE", ["z_start", "z_end"])


class ZSurvey:
    """Takes a Zephyr graph or DWaveSampler with Zephyr topology and
    initializes a survey of the existing/missing nodes, edges. Also, gives a survey of
    external paths.

        Args:
            G (nx.Graph | DWaveSampler): A graph or DWaveSampler with Zephyr topology
    Example:
    >>> from dwave.system import DWaveSampler
    >>> from minorminer.utils.zephyr.zephyr_survey import Survey
    >>> sampler = DWaveSampler(solver="Advantage2_prototype2.6", profile='defaults')
    >>> survey = Survey(sampler)
    >>> print(f"Number of missing nodes is {survey.num_missing_nodes}")
    Number of missing nodes is 33
    >>> print(f"Number of missing edges with both endpoints present is {survey.num_extra_missing_edges}")
    Number of missing edges with both endpoints present is 18
    """

    def __init__(
        self,
        G: nx.Graph | DWaveSampler,
    ) -> None:

        self._shape, self._input_coord_type = self.get_shape_coord(G)
        if isinstance(G, nx.Graph):
            G_nodes = list(G.nodes())
            G_edges = list(G.edges())
        elif isinstance(G, DWaveSampler):
            G_nodes = G.nodelist
            G_edges = G.edgelist
        self._nodes: set[ZNode] = {
            ZNode(coord=ZephyrCoord(*self._input_coord_to_coord(v)), shape=self.shape)
            for v in G_nodes
        }
        self._edges: set[ZEdge] = {
            ZEdge(
                ZNode(coord=ZephyrCoord(*self._input_coord_to_coord(u)), shape=self.shape),
                ZNode(coord=ZephyrCoord(*self._input_coord_to_coord(v)), shape=self.shape),
            )
            for (u, v) in G_edges
        }

    @staticmethod
    def get_shape_coord(G: nx.Graph | DWaveSampler) -> dict[str, ZShape | str]:
        """Returns the shape, coordinates of G, which must be a zephyr graph or
        DWaveSampler with zephyr topology.
        """

        def graph_shape_coord(G: nx.Graph) -> dict[str, int | str]:
            G_info = G.graph
            G_top = G_info.get("family")
            if G_top != "zephyr":
                raise ValueError(f"Expected a graph with zephyr topology, got {G_top}")
            m, t, coord = G_info.get("rows"), G_info.get("tile"), G_info.get("labels")
            return ZShape(m=m, t=t), coord

        def sampler_shape_coord(sampler: DWaveSampler) -> dict[str, int | str]:
            sampler_top: dict[str, str | int] = sampler.properties.get("topology")
            if sampler_top.get("type") != "zephyr":
                raise ValueError(f"Expected a sampler with zephyr topology, got {sampler_top}")
            nodes: list[int] = sampler.nodelist
            edges: list[tuple[int]] = sampler.edgelist
            for v in nodes:
                if not isinstance(v, int):
                    raise NotImplementedError(
                        f"This is implemented only for nodelist containing 'int' elements , got {v}"
                    )
            for e in edges:
                if not isinstance(e, tuple):
                    raise NotImplementedError(
                        f"This is implemented only for edgelist containing 'tuple' elements, got {e}"
                    )
                if len(e) != 2:
                    raise ValueError(f"Expected tuple of length 2 in edgelist, got {e}")
                if not isinstance(e[0], int) or not isinstance(e[1], int):
                    raise NotImplementedError(
                        f"This is implemented only for 'tuple[int]' edgelist, got {e}"
                    )
            coord: str = "int"
            return ZShape(*sampler_top.get("shape")), coord

        if isinstance(G, nx.Graph):
            return graph_shape_coord(G)
        if isinstance(G, DWaveSampler):
            return sampler_shape_coord(G)
        else:
            raise TypeError(f"Expected G to be networkx.Graph or DWaveSampler, got {type(G)}")

    @cached_property
    def _input_coord_to_coord(self) -> Callable[[int | tuple[int]], tuple[int]]:
        """Returns a function that converts the linear or zephyr coordinates to
        the corresponding zephyr coordinates"""
        if self._input_coord_type == "int":
            return dnx.zephyr_coordinates(m=self.shape.m, t=self.shape.t).linear_to_zephyr
        elif self._input_coord_type == "coordinate":
            return lambda v: v
        else:
            raise ValueError(
                f"Expected 'int' or 'coordinate' for self.coord, got {self._input_coord_type}"
            )

    @property
    def shape(self) -> ZShape:
        """Returns the ZShape of G"""
        return self._shape

    @property
    def nodes(self) -> set[ZNode]:
        """Returns the ZNodes of the sampler/graph"""
        return self._nodes

    @cached_property
    def missing_nodes(self) -> set[ZNode]:
        """Returns the ZNodes of the sampler/graph which are missing compared to perfect yield
        Zephyr graph on the same shape.
        """
        parent_nodes = [
            ZNode(coord=ZephyrCoord(*v), shape=self._shape)
            for v in dnx.zephyr_graph(m=self._shape.m, t=self._shape.t, coordinates=True).nodes()
        ]
        return {v for v in parent_nodes if not v in self._nodes}

    @property
    def edges(self) -> set[ZEdge]:
        """Returns the ZEdges of the sampler/graph"""
        return self._edges

    @cached_property
    def missing_edges(self) -> set[ZEdge]:
        """Returns the ZEdges of the sampler/graph which are missing compared to
        perfect yield Zephyr graph on the same shape."""
        parent_edges = [
            ZEdge(ZNode(coord=u, shape=self.shape), ZNode(coord=v, shape=self.shape))
            for (u, v) in dnx.zephyr_graph(
                m=self._shape.m, t=self._shape.t, coordinates=True
            ).edges()
        ]
        return {e for e in parent_edges if not e in self._edges}

    @property
    def extra_missing_edges(self) -> set[ZEdge]:
        """Returns the ZEdges of the sampler/graph which are missing compared to
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

    def neighbors(
        self,
        v: ZNode,
        nbr_kind: str | None = None,
    ) -> set[ZNode]:
        """Returns the neighbours of v when restricted to nbr_kind"""
        if not isinstance(v, ZNode):
            raise TypeError(f"Expected v to be ZNode, got {v}")
        if v in self.missing_nodes:
            return {}
        return {v_nbr for v_nbr in v.neighbors(nbr_kind=nbr_kind) if ZEdge(v, v_nbr) in self._edges}

    def incident_edges(
        self,
        v: ZNode,
        nbr_kind: str | None = None,
    ) -> set[ZEdge]:
        """Returns the edges incident with v when restricted to nbr_kind"""
        nbrs = self.neighbors(v, nbr_kind=nbr_kind)
        if len(nbrs) == 0:
            return set()
        return {ZEdge(v, v_nbr) for v_nbr in nbrs}

    def degree(
        self,
        v: ZNode,
        nbr_kind: str | None = None,
    ) -> int:
        """Returns the degree of v when restricted to nbr_kind"""
        return len(self.neighbors(v, nbr_kind=nbr_kind))

    def _ext_path(self, uwkj: UWKJ) -> set[ZSE]:
        """
        Returns uwkj_sur, where uwkj_sur contains ZSE(z_start, z_end)
        for each non-overlapping external path segment of uwkj.
        """
        z_vals = list(range(self.shape.m))  # As in zephyr coordinates

        def _ext_seg(z_start: int) -> ZSE | None:
            """
            If (u, w, k, j, z_start) does not exist, returns None.
            Else, finds the external path segment starting at z_start going to right, and
            returns the endpoints of the segment (z_start, z_end).
            """
            cur: ZNode = ZNode(coord=ZephyrCoord(*uwkj, z_start), shape=self.shape)
            if cur in self.missing_nodes:
                return None
            upper_z: int = z_vals[-1]
            if z_start > upper_z:
                return None
            if z_start == upper_z:
                return ZSE(z_start, z_start)
            next_ext = ZNode(coord=ZephyrCoord(*uwkj, z_start + 1), shape=self.shape)
            ext_edge = ZEdge(cur, next_ext)
            if ext_edge in self.missing_edges:
                return ZSE(z_start, z_start)
            is_extensible = _ext_seg(z_start + 1)
            if is_extensible is None:
                return ZSE(z_start, z_start)
            return ZSE(z_start, is_extensible.z_end)

        uwkj_sur: set[ZSE] = set()
        while z_vals:
            z_start = z_vals[0]
            seg = _ext_seg(z_start=z_start)
            if seg is None:
                z_vals.remove(z_start)
            else:
                uwkj_sur.add(seg)
                for i in range(seg.z_start, seg.z_end + 1):
                    z_vals.remove(i)

        return uwkj_sur

    def calculate_external_paths_stretch(self) -> dict[UWKJ, set[ZSE]]:
        """
        Returns {uwkj: set of connected segments (z_start, z_end)}
        """
        uwkj_vals = [
            UWKJ(u=u, w=w, k=k, j=j)
            for u in range(2)  # As in zephyr coordinates
            for w in range(2 * self.shape.m + 1)  # As in zephyr coordinates
            for k in range(self.shape.t)  # As in zephyr coordinates
            for j in range(2)  # As in zephyr coordinates
        ]
        return {uwkj: self._ext_path(uwkj) for uwkj in uwkj_vals}
