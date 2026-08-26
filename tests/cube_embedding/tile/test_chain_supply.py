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

from minorminer.utils.zephyr.node_edge import Edge

from minorminer.cube_embedding._tile.chain_supply import (
    find_used_supply,
    generate_chain_supply,
    max_supply_chain,
    prune_zeros,
)


class TestMaxSupplyChain(TestCase):
    def test_no_used_supply(self):
        chain = Edge(0, 1)
        for num in [0, 1, 4]:
            initial_supply = {0: num, 1: num + 1, 2: 100, 3: 0}
            self.assertEqual(max_supply_chain(chain=chain, initial_supply=initial_supply), num)

    def test_used_supply(self):
        chain = Edge(0, 1)
        for num in [1, 4]:
            for num_used in range(num):
                initial_supply = {0: num, 1: num + 1, 2: 100, 3: 0}
                used_supply = {0: num_used, 1: num_used}
                self.assertEqual(
                    max_supply_chain(
                        chain=chain, initial_supply=initial_supply, used_supply=used_supply
                    ),
                    num - num_used,
                )


class TestFindUsedSupply(TestCase):
    def test_basic(self):
        for num_01 in [0, 1, 5]:
            for num_02 in [0, 1, 3]:
                for num_13 in [0, 1, 100]:
                    chain_freq = {Edge(0, 1): num_01, Edge(0, 2): num_02, Edge(1, 3): num_13}
                    idx_freq = {0: num_01 + num_02, 1: num_01 + num_13, 2: num_02, 3: num_13}
                    self.assertEqual(find_used_supply(chain_freq=chain_freq), idx_freq)


class TestPruneZeros(TestCase):
    def test_basic(self):
        for num_01 in [1, 5]:
            for num_02 in [1, 3]:
                for num_13 in [1, 100]:
                    pruned = {Edge(0, 1): num_01, Edge(0, 2): num_02, Edge(1, 3): num_13}
                    chain_freq = pruned | {Edge(4, 5): 0, Edge(5, 6): 0, Edge(0, 3): 0}
                    self.assertEqual(prune_zeros(chain_freq=chain_freq), pruned)


class TestGenerateChainSupply(TestCase):
    def test_basic(self):
        phase_chains = {
            1: [Edge(0, 2), Edge(1, 3), Edge(4, 6), Edge(5, 7)],
            2: [Edge(3, 4)],
            3: [Edge(2, 4), Edge(3, 5)],
            4: [Edge(0, 3), Edge(4, 7)],
        }
        all_edges = [e for list_phase in phase_chains.values() for e in list_phase]
        initial_supply = {i: 2 for e in all_edges for i in e}
        for chain_supply in generate_chain_supply(
            initial_supply=initial_supply, phase_chains=phase_chains, num=1
        ):
            for e in chain_supply:
                self.assertTrue(e in all_edges)

    def test_generates_all_basic(self):
        phase_chains = {
            1: [Edge(0, 2), Edge(1, 3), Edge(4, 6), Edge(5, 7)],
            2: [Edge(3, 4)],
            3: [Edge(2, 4), Edge(3, 5)],
            4: [Edge(0, 3), Edge(4, 7)],
        }
        all_edges = {e for list_phase in phase_chains.values() for e in list_phase}

        initial_supply = {i: 1 for e in all_edges for i in e}
        generated_chains = set()
        for dict_edge_freq in generate_chain_supply(
            initial_supply=initial_supply, phase_chains=phase_chains, num=1
        ):
            self.assertEqual(len(dict_edge_freq), 1)
            generated_chains.add(list(dict_edge_freq.keys())[0])

        self.assertEqual(all_edges, generated_chains)

    def test_generates_all(self):
        phase_chains = {
            1: [Edge(0, 2), Edge(1, 3), Edge(4, 6), Edge(5, 7)],
            2: [Edge(3, 4)],
            3: [Edge(2, 4), Edge(3, 5)],
            4: [Edge(0, 3), Edge(4, 7)],
        }
        for num_02 in [1, 2]:
            for num_35 in [1, 3]:
                initial_supply = {0: num_02, 2: num_02, 3: num_35, 5: num_35}
                all_generated = generate_chain_supply(
                    initial_supply=initial_supply, phase_chains=phase_chains, num=num_02 + num_35
                )
                self.assertEqual(all_generated, [{Edge(0, 2): num_02, Edge(3, 5): num_35}])

        num = 3
        initial_supply = {0: num, 2: num, 3: num, 5: num}
        all_generated = generate_chain_supply(
            initial_supply=initial_supply, phase_chains=phase_chains, num=num
        )
        for i in range(num + 1):
            self.assertTrue(prune_zeros({Edge(0, 2): i, Edge(3, 5): num - i}) in all_generated)

    def test_num_phases(self):
        phase_chains = {
            1: [Edge(0, 2), Edge(1, 3), Edge(4, 6), Edge(5, 7)],
            2: [Edge(3, 4)],
            3: [Edge(2, 4), Edge(3, 5)],
            4: [Edge(0, 3), Edge(4, 7)],
        }
        all_edges = [e for list_phase in phase_chains.values() for e in list_phase]
        initial_supply = {i: 2 for e in all_edges for i in e}
        for num_phases in [1, 2, 3, 4]:
            edges_until_phase_num = [
                e for i, list_phase in phase_chains.items() for e in list_phase if i <= num_phases
            ]
            for chain_supply in generate_chain_supply(
                initial_supply=initial_supply,
                phase_chains=phase_chains,
                num=1,
                num_phases=num_phases,
            ):
                for e in chain_supply:
                    self.assertTrue(e in edges_until_phase_num)
