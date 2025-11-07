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


from unittest import TestCase

from dwave_networkx import zephyr_graph
from minorminer.utils.zephyr.survey import ZSE

from minorminer._lattice_utils.survey import ZLatticeSurvey


class TestLatticeSurvey(TestCase):
    def test_num_missing_nodes(self):
        for m in [1, 3, 6]:
            G = zephyr_graph(m=m, t=4)
            for num_missing in [0, 1, 10]:
                G.remove_nodes_from(list(range(num_missing)))
                lsurvey = ZLatticeSurvey(G)
                self.assertEqual(lsurvey.num_missing_nodes, num_missing)

    def test_num_missing_edges(self):
        for m in [1, 3, 6]:
            G = zephyr_graph(m=m, t=4)
            G_edges = list(G.edges())
            for num_missing in [0, 1, 10]:
                G.remove_edges_from(G_edges[:num_missing])
                lsurvey = ZLatticeSurvey(G)
                self.assertEqual(lsurvey.num_missing_edges, num_missing)

    def test_shape(self):
        for m in [1, 3, 6]:
            for t in [1, 4, 6]:
                G = zephyr_graph(m=m, t=t)
                lsurvey = ZLatticeSurvey(G=G)
                self.assertEqual((lsurvey.m, lsurvey.t), (m, t))

    def test_calculate_external_paths_stretch(self):
        for m in [3, 6]:
            whole_stretch = [ZSE(z_start=0, z_end=m - 1)]
            G = zephyr_graph(m=m, t=4, coordinates=True)
            u_vals = (0, 1)
            w_vals = (0, m, 2 * m)
            k_vals = (0, 3)
            j_val, z_val = 0, 1
            nodes_to_remove = [
                (u, w, k, j_val, z_val) for u in u_vals for w in w_vals for k in k_vals
            ]
            G.remove_nodes_from(nodes_to_remove)
            lsurvey_external_paths_stretch = ZLatticeSurvey(G).calculate_external_paths_stretch()
            for uwj, uwj_dict in lsurvey_external_paths_stretch.items():
                u, w, j = uwj
                for k, z_stretches in uwj_dict.items():
                    if (u, w, k, j, z_val) in nodes_to_remove:
                        self.assertNotEqual(z_stretches, whole_stretch)
                    else:
                        self.assertEqual(z_stretches, whole_stretch)
