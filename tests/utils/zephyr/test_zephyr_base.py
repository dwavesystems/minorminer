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


import unittest
from itertools import product
import numpy as np
from dwave.cloud import Client
from dwave.system import DWaveSampler
import dwave_networkx as dnx

class ZephyrBaseTest(unittest.TestCase):
    def setUp(self):
        self.initialize_samplers()
        self.initialize_rngs()
        self.initialize_node_del_percents()
        self.initialize_edge_del_percents()
        self.initialize_zeph_ms()
        self.initialize_graphs()

    def initialize_samplers(self):
        with Client.from_config(client="qpu") as client:
            zeph_solvers = client.get_solvers(
                topology__type__eq="zephyr",
                )
        self.samplers = [DWaveSampler(solver=z_solver.id) for z_solver in zeph_solvers]

    def initialize_rngs(self):
        self.rngs = [
            np.random.default_rng(seed=1),
            np.random.default_rng(seed=10),
        ]
        
    def initialize_zeph_ms(self):
        self.zeph_ms = [3, 6]

    def initialize_node_del_percents(self):
        self.node_del_percents = [0, 0.03]


    def initialize_edge_del_percents(self):
        self.edge_del_percents = [0, 0.02]


    def initialize_graphs(self):
        self.graphs = list()
        for rng in self.rngs:
            for m in self.zeph_ms:
                for node_del_per, edge_del_per in product(
                    self.node_del_percents, self.edge_del_percents
                    ):
                    G = dnx.zephyr_graph(m=m, coordinates=True)
                    num_nodes_to_remove = int(node_del_per * G.number_of_nodes())
                    nodes_to_remove = [
                        tuple(v)
                        for v in rng.choice(G.nodes(), num_nodes_to_remove)
                        ]
                    G.remove_nodes_from(nodes_to_remove)
                    num_edges_to_remove = int(edge_del_per * G.number_of_edges())
                    edges_to_remove = [
                        (tuple(u), tuple(v))
                        for (u, v) in rng.choice(G.edges(), num_edges_to_remove)
                        ]
                    G.remove_edges_from(edges_to_remove)
                    G_dict = {
                        "G": G,
                        "m": m,
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                        "num_missing_nodes": num_nodes_to_remove,
                        "num_missing_edges": dnx.zephyr_graph(m=m).number_of_edges()-G.number_of_edges(),
                        "num_extra_missing_edges": num_edges_to_remove,
                        }
                    self.graphs.append(G_dict)