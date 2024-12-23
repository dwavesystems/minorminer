"""
This demo illustrates the process of checking graph embedding feasibility and performing raster embeddings
across various D-Wave graph topologies such as Chimera, Pegasus, and Zephyr.

The demo performs the following steps:

1. **Embedding Feasibility Checks**:
    - For each source topology, it generates a subgraph and checks whether it can be embedded into each target topology.
    - It reports whether the embedding is feasible and, if so, the required sublattice size.

2. **Raster Embedding Examples**:
    - For each topology, it performs raster embeddings at the minimal scale.
    - It visualizes the embeddings if visualization is enabled.
    - It validates the embeddings and displays the results.

This example demonstrates how to assess and visualize graph embeddings using D-Wave's `dwave_networkx` library.

"""

import dwave_networkx as dnx

from minorminer.utils.parallel_embeddings import (
    find_sublattice_embeddings,
    embeddings_to_array,
)

from minorminer.utils.feasibility import (
    lattice_size_lower_bound,
)


def main():
    print("Minimum Subgraph Embedding (graph rows) Examples\n")

    # Parameters
    visualize = True  # Enable or disable visualization
    topologies = ["chimera", "pegasus", "zephyr"]  # Target graph topologies
    smallest_tile = {
        "chimera": 1,
        "pegasus": 2,
        "zephyr": 1,
    }  # Minimum tile sizes for each topology
    generators = {
        "chimera": dnx.chimera_graph,
        "pegasus": dnx.pegasus_graph,
        "zephyr": dnx.zephyr_graph,
    }

    # Iterate over each source topology for embedding feasibility checks
    for source_topology in topologies:
        sublattice_size_S = smallest_tile[source_topology] + 1
        S = generators[source_topology](sublattice_size_S)

        print(
            f"Evaluating embeddings for source topology '{source_topology}' with sublattice size {sublattice_size_S}."
        )

        # Check embedding feasibility into each target topology
        for target_topology in topologies:
            # Check one-to-one embedding feasibility
            sublattice_size = lattice_size_lower_bound(
                S, topology=target_topology, one_to_one=True
            )
            if sublattice_size is None:
                print(
                    f"Embedding {source_topology}-{sublattice_size_S} into "
                    f"{target_topology} is infeasible."
                )
            else:
                print(
                    f"Embedding {source_topology}-{sublattice_size_S} into "
                    f"{target_topology} may be feasible, requires sublattice size >= {sublattice_size}."
                )

            # Check embedding feasibility with specific target graph
            T = generators[target_topology](smallest_tile[target_topology])
            sublattice_size = lattice_size_lower_bound(S, T=T)
            if sublattice_size is None:
                print(
                    f"Embedding {source_topology}-{sublattice_size_S} into "
                    f"{target_topology}-{smallest_tile[target_topology]} is infeasible."
                )
            else:
                print(
                    f"Embedding {source_topology}-{sublattice_size_S} into "
                    f"{target_topology}-{smallest_tile[target_topology]} may be feasible, "
                    f"requires sublattice size >= {sublattice_size}."
                )
        print()

    print("Raster Embedding Examples\n")

    # Perform raster embedding examples for each topology
    for topology in topologies:
        min_raster_scale = smallest_tile[topology]
        S = generators[topology](min_raster_scale)
        T = generators[topology](min_raster_scale + 1)  # Allows for rastering

        print(f"\nTopology: {topology}")

        # Perform Embedding Search and Validation
        embs = find_sublattice_embeddings(
            S, T, sublattice_size=min_raster_scale, max_num_emb=float("inf")
        )
        print(f"{len(embs)} independent embeddings found by rastering.")
        print(embs)

        # Validate embeddings
        assert all(
            set(emb.keys()) == set(S.nodes()) for emb in embs
        ), "Mismatch in source nodes."
        assert all(
            set(emb.values()).issubset(set(T.nodes())) for emb in embs
        ), "Mismatch in target nodes."
        value_list = [v for emb in embs for v in emb.values()]
        assert len(set(value_list)) == len(
            value_list
        ), "Duplicate target nodes in embeddings."

        # Visualize embeddings if enabled (waiting for visualization function to be migrated to dwave networkx)
        # if visualize:
        #     plt.figure(figsize=(12, 12))
        #     visualize_embeddings(T, embeddings=embs)

        #     # Create a subgraph of S with a limited number of edges for visualization
        #     S_aux = nx.Graph()
        #     S_aux.add_nodes_from(S)
        #     S_aux.add_edges_from(list(S.edges)[:10])  # First 10 edges only
        #     plt.figure(figsize=(12, 12))
        #     visualize_embeddings(T, embeddings=embs, S=S_aux)
        #     plt.show()

        # Perform direct search for embeddings
        embs_direct = find_sublattice_embeddings(S, T)
        print(f"{len(embs_direct)} independent embeddings found by direct search.")
        assert all(
            set(emb.keys()) == set(S.nodes()) for emb in embs_direct
        ), "Mismatch in source nodes."
        assert all(
            set(emb.values()).issubset(set(T.nodes())) for emb in embs_direct
        ), "Mismatch in target nodes."
        value_list_direct = [v for emb in embs_direct for v in emb.values()]
        assert len(set(value_list_direct)) == len(
            value_list_direct
        ), "Duplicate target nodes in direct embeddings."

        print("Default embeddings (full graph search) presented as an ndarray:")
        print(embeddings_to_array(embs_direct, as_ndarray=True))
        print()

    print("Demo completed. See additional usage examples in test_embeddings.")


if __name__ == "__main__":
    main()
