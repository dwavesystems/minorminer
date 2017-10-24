minorminer
==========

minorminer is a heuristic tool for finding graph minors. Given a possible minor and a target graph, it tries to find a mapping that embeds the minor into the target graph.

The primary utility is a function called ```find_embedding```, which is an implementation of the algorithm described in [1].  Our implementation of this heuristic algorithm accepts various optional parameters, which are used to tune the algorithm's execution or constrain the problem in consideration.

The twin goal of this implementation is to provide enough hooks to the user that it will be easy to use this code as a basic building block in future research, without any degradation in performance when compared to well-tuned implementation of the algorithm which omits those parameters.

[1] https://arxiv.org/abs/1406.2741


Getting Started
---------------

### Python

```cd python; python setup.py install; python -m nose .. --exe```

```pip install minorminer```

**TODO** Usage example?

### Matlab

**TODO** Explain how to build or use the matlab package.

**TODO** Usage example?

### C++

**TODO** Should we include a CMake file to make it easier to use this package as a c++ library?

**TODO** Usage example?
