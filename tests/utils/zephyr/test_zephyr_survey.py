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


from minorminer.utils.zephyr.survey import ZSurvey
from tests.utils.zephyr.test_zephyr_base import ZephyrBaseTest


class TestZephyrSurvey(ZephyrBaseTest):
    def setUp(self) -> None:
        super().setUp()

    def test_get_zephyr_shape_coord_sampler(self) -> None:
        for z_sampler in self.samplers:
            (m, t), coord = ZSurvey(z_sampler).get_shape_coord(z_sampler)
            sampler_top = z_sampler.properties["topology"]
            self.assertTrue(m, sampler_top.get("shape")[0])
            self.assertTrue(t, sampler_top.get("shape")[1])
            if coord == "int":
                for v in z_sampler.nodelist:
                    self.assertTrue(isinstance(v, int))
            elif coord == "coordinates":
                for v in z_sampler.nodelist:
                    self.assertTrue(isinstance(v, tuple))

    def test_get_zephyr_shape_coord_graph(self) -> None:
        for G_dict in self.graphs:
            z_graph = G_dict["G"]
            (m, t), coord = ZSurvey(z_graph).get_shape_coord(z_graph)
            G_info = z_graph.graph
            self.assertTrue(m, G_info.get("rows"))
            self.assertTrue(t, G_info.get("tile"))
            self.assertTrue(coord, G_info.get("labels"))

    def test_zephyr_survey_runs_sampler(self) -> None:
        for z_sampler in self.samplers:
            try:
                ZSurvey(z_sampler)
            except Exception as e:
                self.fail(
                    f"ZephyrSurvey raised an exception {e} when running with sampler = {z_sampler}"
                )

    def test_zephyr_survey_runs_graph(self) -> None:
        for G_dict in self.graphs:
            z_graph = G_dict["G"]
            try:
                ZSurvey(z_graph)
            except Exception as e:
                self.fail(
                    f"ZephyrSurvey raised an exception {e} when running with graph = {z_graph}"
                )

    def test_num_nodes(self) -> None:
        for z_sampler in self.samplers:
            num_nodes = len(z_sampler.nodelist)
            self.assertEqual(len(ZSurvey(z_sampler).nodes), num_nodes)
        for G_dict in self.graphs:
            z_graph = G_dict["G"]
            num_nodes = G_dict["num_nodes"]
            self.assertEqual(len(ZSurvey(z_graph).nodes), num_nodes)

    def test_num_edges(self) -> None:
        for z_sampler in self.samplers:
            num_edges = len(z_sampler.edgelist)
            self.assertEqual(len(ZSurvey(z_sampler).edges), num_edges)
        for G_dict in self.graphs:
            z_graph = G_dict["G"]
            num_edges = G_dict["num_edges"]
            self.assertEqual(len(ZSurvey(z_graph).edges), num_edges)

    def test_external_paths(self) -> None:
        for z_sampler in self.samplers:
            sur = ZSurvey(z_sampler)
            sur.calculate_external_paths_stretch()
