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
pip install -r requirements.txt
python setup.py install
# optionally, run the tests to check your build
pip install -r ../tests/requirements.txt
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

The mex bindings for this library will work with some versions of 2013 and earlier,
and versions from 2016b an onward. An example build command used in Ubuntu is
found in the makefile `matlab/make.m`.

If you run `make` in the `matlab` directory on Ubuntu it should generate
`find_embedding.mexa64`, which can be added to the MATLAB path.

#### Examples
```Matlab
% A triangle is a minor of a square.
triangle = triu(ones(3),1);
square = sparse([1,2,3,4],[2,3,4,1],[1,1,1,1],4,4);

% Find an assignment of sets of square variables to the triangle variables
options = struct('random_seed',10);
embedding = find_embedding_matlab_wrapper(triangle, square, options)
% typically in matlab we use indices starting at one rather than 0:
embedding = cellfun(@(x)x+1,embedding,'UniformOutput',false);
embedding{:}
```

```Matlab
% We can insist that variable 0 of the triangle will always be assigned to
% [2] (zero-indexed)
chains = cell(1);
chains{1} = 2;
options = struct();
options.fixed_chains = chains;
embedding = find_embedding(triangle, square, options)
embedding{:}
```

```Matlab
% If we didn't want to force variable 0 to stay as [2], but we thought that
% was a good start we could provide it as an initialization hint instead.
options = struct();
options.initial_chains = chains;
embedding = find_embedding(triangle, square, options)
embedding{:}
```

### C++

#### Installation

The `CMakeLists.txt` in the root of this repo will build the library and run a series of tests. On linux the commands would be something like this:

```
mkdir build; cd build
cmake ..
make
```

#### Library Usage

C++11 programs should be able to use this as a header only library. If your project is using CMake this library can be used fairly simply; if you have checked out this repo as `externals/minorminer` in your project you would need to add the following lines to your `CMakeLists.txt`

```CMake
set(MINORMINER_BUILD_TESTS OFF)
add_subdirectory(externals/minorminer)

# After your target is defined
target_link_libraries(your_target minorminer pthread)
```

#### Examples

A minimal example that can be built in the root of this repo as `example.cpp`.

```bash
g++ example.cpp -std=c++11 -o example -pthread
```

```c++
#include "include/find_embedding.hpp"
#include <iostream>

class MyCppInteractions : public find_embedding::LocalInteraction {
public:
    bool _canceled = false;
    void cancel() { _canceled = true; }

private:
    virtual void displayOutputImpl(const std::string& mess) const {
        std::cout << mess << std::endl;
    }
    virtual bool cancelledImpl() const {
        return _canceled;
    }
};

int main(){

    graph::input_graph triangle(3, {0, 1, 2}, {1, 2, 0});
    graph::input_graph square(4, {0, 1, 2, 3}, {1, 2, 3, 0});
    find_embedding::optional_parameters params;
    params.localInteractionPtr.reset(new MyCppInteractions());

    std::vector<std::vector<int>> chains;

    if(find_embedding::findEmbedding(triangle, square, params, chains)) {
        for(auto chain : chains){
            for(auto var : chain)
                std::cout << var << " ";
            std::cout << std::endl;
        }
    } else {
        std::cout << "Couldn't find embedding." << std::endl;
    }

    return 0;
}
```
