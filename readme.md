minorminer
==========

minorminer is a heuristic tool for finding graph minors. Given a possible minor and a target graph, it tries to find a mapping that embeds the minor into the target graph.

The primary utility is a function called ```find_embedding```, which is an implementation of the algorithm described in [1].  Our implementation of this heuristic algorithm accepts various optional parameters, which are used to tune the algorithm's execution or constrain the problem in consideration.

The twin goal of this implementation is to provide enough hooks to the user that it will be easy to use this code as a basic building block in future research, without any degradation in performance when compared to well-tuned implementation of the algorithm which omits those parameters.

[1] https://arxiv.org/abs/1406.2741


Getting Started
---------------

### Python

#### Installation
If a wheel for your platform has been precompiled and posted to pypi
installing it with pip is recommended.

```bash
pip install minorminer
```

If your platform doesn't have a precompiled wheel, try to run the `setuptools` script
in the python directory.

```bash
python setup.py install
# optionally, install nose and run the tests to check your build
pip install nose
python -m nose .. --exe
```

#### Examples
```python
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
```
```python
# We can insist that variable 0 of the triangle will always be assigned to [2]
embedding = find_embedding(triangle, square, fixed_chains={0: [2]})
print(embedding)
# [[2], [3, 0], [1]]
# [[2], [1], [0, 3]]
# And more, but all of them start with [2]
```
```python
# If we didn't want to force variable 0 to stay as [2], but we thought that
# was a good start we could provide it as an initialization hint instead.
embedding = find_embedding(triangle, square, initial_chains={0: [2]})
print(embedding)
# [[2], [0, 3], [1]]
# [[0], [3], [1, 2]]
# Output where variable 0 has switched to something else is possible again.
```
```python
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
```

### Matlab

#### Installation
**TODO** Explain how to build or use the matlab package.

#### Examples
**TODO** Usage example?

### C++

#### Installation
**TODO** Should we include a CMake file to make it easier to use this package as a c++ library?

#### Examples
**TODO** Usage example?
