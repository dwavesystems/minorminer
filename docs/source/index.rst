.. _index_minorminer:

==========
minorminer
==========

.. toctree::
    :caption: Reference documentation for minorminer:
    :maxdepth: 1

    reference/api_ref

About minorminer
================

.. include:: README.rst
    :start-after: start_minorminer_about
    :end-before: end_minorminer_about

Examples
--------

.. include:: README.rst
    :start-after: start_minorminer_examples_python
    :end-before: end_minorminer_examples_python

.. todo:: update this entire section (taken from the intro)

Introduction
------------

`minorminer` is a library of tools for finding graph minor embeddings, developed
to embed Ising problems onto quantum annealers (QA). While this library can be
used to find minors in arbitrary graphs, it is particularly geared towards
state-of-the-art QA: problem graphs of a few to a few hundred variables, and
hardware graphs of a few thousand qubits.

`minorminer` has both a Python and C++ API, and includes implementations of
multiple embedding algorithms to best fit different problems.

Minor-Embedding and QPU Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an introduction to minor-embedding, see the :ref:`qpu_embedding_intro`
section.

For an introduction to the topologies of D-Wave hardware graphs, see the
:ref:`qpu_topologies` section and the
`Exploring Pegasus Jupyter Notebook <https://github.com/dwave-examples/pegasus-notebook>`_
that explains the :term:`Advantage` architecture in further detail.

Minor-embedding can be done manually, though typically for very small problems
only. For a walkthrough of the manual minor-embedding process, see the
:ref:`qpu_example_sat_constrained` section.

Minor-Embedding in Ocean
~~~~~~~~~~~~~~~~~~~~~~~~

Minor-embedding can also be automated through Ocean. `minorminer` is used by
several :ref:`Ocean embedding composites <system_composites>` for this purpose.
For details on automated (and manual) minor-embedding through
Ocean, see how the :class:`~dwave.system.composites.EmbeddingComposite` and
:class:`~dwave.system.composites.FixedEmbeddingComposite` classes are used
in this :ref:`Boolean AND Gate example <qpu_example_and>`.

Once an embedding has been found, D-Wave's Problem Inspector tool can be used to
evaluate its quality. See the :ref:`qpu_example_inspector_graph_partitioning`
section for more information.

Usage Information
=================

*   :ref:`index_concepts` for terminology
*   :ref:`qpu_embedding_intro` for an introduction to minor embedding
*   :ref:`qpu_embedding_guidance` provides advanced guidance
*   Examples in the :ref:`qpu_example_and`, :ref:`qpu_example_multigate`,
    and :ref:`qpu_example_inspector_graph_partitioning` sections show through
    some simple examples how to embed and set chain strength.