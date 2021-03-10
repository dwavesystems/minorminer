.. intro_minorminer:

============
Introduction
============

.. automodule:: minorminer

`minorminer` is a library of tools for finding graph minor embeddings, developed 
to embed Ising problems onto quantum annealers (QA). For an introduction to 
minor-embedding, see :std:doc:`Minor-Embedding <oceandocs:concepts/embedding>`. 

While this library can be used to find minors in arbitrary graphs, it is 
particularly geared toward state-of-the-art QA: problem graphs of a few to a few 
hundred variables, and hardware graphs of a few thousand qubits. For an 
introduction to the topologies of D-Wave hardware graphs, see 
:std:doc:`QPU Topology <oceandocs:concepts/topology>`.

Minor-embedding can be done manually, though typically for very small problems 
only. For a walkthrough of the manual minor-embedding process, see the 
`Constraints Example: Minor-Embedding <https://docs.dwavesys.com/docs/latest/c_gs_7.html>`_. 

Minor-embedding can also be automated through Ocean. `minorminer` is used by several
:std:doc:`Ocean embedding composites <oceandocs:docs_system/reference/composites>`
for this purpose. For details on automated (and manual) minor-embedding through 
Ocean, see how the `EmbeddingComposite` and `FixedEmbeddingComposite` are used 
in this :std:doc:`Boolean AND Gate example <oceandocs:examples/and>`. 

Once an embedding has been found, D-Wave's Problem Inspector tool can be used to 
evaluate its quality. See
:std:doc:`Using the Problem Inspector <oceandocs:examples/inspector_graph_partitioning>`
for more information.
