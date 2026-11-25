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


"""
A collection of helpers to calculate possible chain allocations, given the supplied
node index frequencies.
"""

from collections import defaultdict
from itertools import product

from minorminer.utils.zephyr.node_edge import Edge

__all__ = []


def max_supply_chain(
    chain: Edge, initial_supply: dict[int, int], used_supply: dict[int, int] | None = None
) -> int:
    """
    Returns the maximum number of times the given ``chain`` can be constructed,
    based on ``initial_supply`` and optionally reduced by ``used_supply``.

    Args:
        chain (Edge): The chain to construct, represented as an ``Edge`` between indices
            of two internal neighbors in a ``QuoTile`` tile.
        initial_supply (dict[int, int]): The available supply for each node index.
        used_supply (dict[int, int] | None, optional): The amount of supply already used for each node index. Defaults to ``None``.

    Returns:
        int: The maximum number of times ``chain`` can be constructed given the current supply constraints.
    """
    used_supply = used_supply or {}
    a, b = chain
    return min(
        initial_supply.get(a, 0) - used_supply.get(a, 0),
        initial_supply.get(b, 0) - used_supply.get(b, 0),
    )


def find_used_supply(chain_freq: dict[Edge, int]) -> dict[int, int]:
    """
    Given a dictionary of chain frequencies, returns a dictionary mapping each node index
    to the total number of times it is used across all chains.

    Args:
        chain_freq (dict[Edge, int]): A mapping from ``Edge(a, b)`` (representing a chain between node indices ``a`` and ``b``)
            to the number of times that chain is used.

    Returns:
        dict[int, int]: A mapping from node index to its total used supply (i.e., total participation in chains).
    """
    used_supply = {}
    for (a, b), freq in chain_freq.items():
        used_supply[a] = used_supply.get(a, 0) + freq
        used_supply[b] = used_supply.get(b, 0) + freq
    return used_supply


def prune_zeros(chain_freq: dict[Edge, int]) -> dict[Edge, int]:
    """Removes entries with zero frequency from a dictionary mapping chains to frequencies.

    Args:
        chain_freq (dict[Edge, int]): The frequency dictionary.

    Returns:
        dict[Edge, int]: The dictionary after removing all items with value zero.
    """
    return {chain: chain_freq for chain, chain_freq in chain_freq.items() if chain_freq != 0}


def generate_chain_supply(
    initial_supply: dict[int, int],
    phase_chains: dict[int, list[Edge]],
    num: int,
    num_phases: int | None = None,
    **kwargs,
) -> list[dict[Edge, int]]:
    """
    Returns the list of all possible chain allocation configurations for a given number of chains to be constructed,
    while respecting initial supply constraints and restricting to the chains per phase.

    Each chain allocation is a dictionary:
        {Edge(idx1, idx2): count, Edge(idx1, idx3): count, ...}
    indicating how many chains of each type can be constructed.

    Args:
        initial_supply: Maps each index (typically the index of a node in a ``QuoTile`` tile)
            to its available supply count in partially-yielded Zephyr.
        phase_chains: Maps each phase number to the list of allowable Edge chains in that phase.
        num: Total number of chains to construct across all phases.
        num_phases: Number of phases to run.

    Returns:
        list[dict[Edge, int]]: A list of dictionaries, each representing a supply allocation for the chains which has ``num`` chains.
    """

    def phase_supply(phase, prev_phases_supp, phase_chains):
        """Returns all possible chain allocation configurations of the given phase ``phase``
        given the already used supply ``prev_phases_supp``."""
        prev_used_supp = find_used_supply(prev_phases_supp)
        max_phase_chains_supply = {
            chain: max_supply_chain(
                chain=chain, initial_supply=initial_supply, used_supply=prev_used_supp
            )
            for chain in phase_chains[phase]
        }

        chains, range_chains = [], []
        for c, c_freq in max_phase_chains_supply.items():
            if c_freq == 0:
                continue
            chains.append(c)
            range_chains.append(range(c_freq + 1))

        return [prune_zeros(dict(zip(chains, values))) for values in product(*range_chains)]

    def get_supply_num(
        list_chains_supplies: list[dict[Edge, int]], num: int
    ) -> list[dict[Edge, int]]:
        """Returns the elements of ``list_chains_supplies`` whose sum of all chain frequencies add up to ``num``."""
        supply_ = defaultdict(list)
        for chains_supply in list_chains_supplies:
            supply_[sum(chains_supply.values())].append(chains_supply)
        return supply_.get(num, [])

    def run_phase(
        prev_phases_supply_options: list[dict[Edge, int]], phase: int
    ) -> list[dict[Edge, int]]:
        """Returns all possible chain allocation configurations of up to the given phase ``phase``."""
        return [
            phase_supp | prev_phases_supp
            for prev_phases_supp in prev_phases_supply_options
            for phase_supp in phase_supply(
                prev_phases_supp=prev_phases_supp, phase=phase, phase_chains=phase_chains
            )
        ]

    if num < 0:
        raise ValueError(f"Expected num to be at least zero, got {num}")

    num_phases = num_phases or max(phase_chains)

    supp_options = [{}]
    for phase in range(1, num_phases + 1):
        supp_options = run_phase(prev_phases_supply_options=supp_options, phase=phase)

    return get_supply_num(
        list_chains_supplies=supp_options,
        num=num,
    )
