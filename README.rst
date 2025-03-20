.. image:: https://img.shields.io/pypi/v/minorminer.svg
    :target: https://pypi.org/project/minorminer

.. image:: https://img.shields.io/pypi/pyversions/minorminer.svg
    :target: https://pypi.python.org/pypi/minorminer

.. image:: https://circleci.com/gh/dwavesystems/minorminer.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/minorminer

.. image:: https://img.shields.io/badge/arXiv-1406.2741-b31b1b.svg
    :target: https://arxiv.org/abs/1406.2741

.. image:: https://img.shields.io/badge/arXiv-1507.04774-b31b1b.svg
    :target: https://arxiv.org/abs/1507.04774


==========
minorminer
==========

.. start_minorminer_about

`minorminer` is a heuristic tool for minor embedding: given a minor and target
graph, it tries to find a mapping that embeds the minor into the target.

.. start_minorminer_about_general_embedding

The primary utility function, ``find_embedding()``, is an implementation of
the heuristic algorithm described in [1]. It accepts various optional parameters
used to tune the algorithm's execution or constrain the given problem.

This implementation performs on par with tuned, non-configurable implementations
while providing users with hooks to easily use the code as a basic building
block in research.

[1] https://arxiv.org/abs/1406.2741

Another function, ``find_clique_embedding()``, can be used to find clique
embeddings for Chimera, Pegasus, and Zephyr graphs in polynomial time. It is an
implementation of the algorithm described in [2]. There are additional utilities
for finding biclique embeddings as well.

[2] https://arxiv.org/abs/1507.04774

.. end_minorminer_about

Python
======

Installation
------------

pip installation is recommended for platforms with precompiled wheels posted to
pypi. Source distributions are provided as well.

.. code-block:: bash

    pip install minorminer

To install from this repository, you will need to first fetch the submodules

    git submodule init
    git submodule update

and then run the `setuptools` script.

.. code-block:: bash


    pip install -r requirements.txt
    python setup.py install
    # optionally, run the tests to check your build
    pip install -r tests/requirements.txt
    python -m pytest .


Examples
--------

.. start_minorminer_examples_python

.. code-block:: python

    from minorminer import find_embedding

    # A triangle is a minor of a square.
    triangle = [(0, 1), (1, 2), (2, 0)]
    square = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Find an assignment of sets of square variables to the triangle variables
    embedding = find_embedding(triangle, square, random_seed=10)
    print(len(embedding))  # 3, one set for each variable in the triangle
    print(embedding)
    # We don't know which variables will be assigned where, here are a
    # couple possible outputs:
    # [[0, 1], [2], [3]]
    # [[3], [1, 0], [2]]

.. code-block:: python

    # We can insist that variable 0 of the triangle will always be assigned to [2]
    embedding = find_embedding(triangle, square, fixed_chains={0: [2]})
    print(embedding)
    # [[2], [3, 0], [1]]
    # [[2], [1], [0, 3]]
    # And more, but all of them start with [2]

.. code-block:: python

    # If we didn't want to force variable 0 to stay as [2], but we thought that
    # was a good start we could provide it as an initialization hint instead.
    embedding = find_embedding(triangle, square, initial_chains={0: [2]})
    print(embedding)
    # [[2], [0, 3], [1]]
    # [[0], [3], [1, 2]]
    # Output where variable 0 has switched to something else is possible again.

.. code-block:: python

    import networkx as nx

    # An example on some less trivial graphs
    # We will try to embed a fully connected graph with 6 nodes, into a
    # random regular graph with degree 3.
    clique = nx.complete_graph(6).edges()
    target_graph = nx.random_regular_graph(d=3, n=30).edges()

    embedding = find_embedding(clique, target_graph)

    print(embedding)
    # There are many possible outputs for this, sometimes it might even fail
    # and return an empty list

A more fleshed out example can be found under `examples/fourcolor.py`

.. code-block:: bash

    cd examples
    pip install -r requirements.txt
    python fourcolor.py

.. end_minorminer_examples_python

C++
===

Installation
------------

The `CMakeLists.txt` in the root of this repo will build the library and
optionally run a series of tests. On Linux, the commands would be something like
this:

.. code-block:: bash

    mkdir build; cd build
    cmake ..
    make

To build the tests, turn the CMake option `MINORMINER_BUILD_TESTS` on. The
command line option for CMake to do this would be `-DMINORMINER_BUILD_TESTS=ON`.

Library Usage
-------------

C++11 programs should be able to use this as a header-only library. If your
project is using CMake, this library can be used fairly simply; if you have
checked out this repo as `externals/minorminer` in your project, you would need
to add the following lines to your `CMakeLists.txt`

.. code-block:: CMake

    add_subdirectory(externals/minorminer)

    # After your target is defined
    target_link_libraries(your_target minorminer pthread)

Examples
--------

A minimal buildable example can be found in this repo under
`examples/example.cpp`.

.. code-block:: bash

    cd examples
    g++ example.cpp -std=c++11 -o example -pthread

This can also be built using the included `CMakeLists.txt` along with the main
library build by turning the CMake option `MINORMINER_BUILD_EXAMPLES` on. The
command line option for CMake to do this would be
`-DMINORMINER_BUILD_EXAMPLES=ON`.

License
=======

Released under the Apache License 2.0. See `<LICENSE>`_ file.

Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.

If you're interested in adding or modifying parameters of the ``find_embedding``
primary utility function, please see the `<parameter_checklist.txt>`_ file.

Release Notes
-------------

``minorminer`` makes use of `reno <https://docs.openstack.org/reno/>`_
to manage its release notes.

When making a contribution to ``minorminer`` that will affect users,
create a new release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``. Remove any sections
not relevant to your changes. Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
