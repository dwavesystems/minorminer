# Copyright 2024 D-Wave Inc.
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

import os
import platform
import sys

from Cython.Build import cythonize
from setuptools import setup, Extension


# Change directories so this works when called from other locations.
# Useful in build systems that build from source.
setup_folder_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_folder_loc)

DEBUG = '--debug' in sys.argv or '-g' in sys.argv or 'CPPDEBUG' in os.environ
SAFE_COORDS = 'SAFE_COORDS' in os.environ

extra_compile_args = []
extra_compile_args_glasgow = []  # specific to glasgow
extra_compile_args_minorminer = []  # specific to minorminer
if platform.system().lower() == "windows":
    extra_compile_args.extend([
        '/std:c++17',
        '/MT',
        '/EHsc',
        ])

    extra_compile_args_glasgow.extend([
        '/external:W4',
        '/external:I external',
        '/DUSE_PORTABLE_SNIPPETS_BUILTIN',
        ])

    extra_compile_args_minorminer.extend([
        '/DCPPDEBUG' if DEBUG else None,
        '/DSAFE_COORDS' if SAFE_COORDS else None,
        ])

else:  # Unix
    extra_compile_args.extend([
        '-std=c++17',
        '-Wall',
        '-Wno-format-security',
        '--O0' if DEBUG else '-Ofast',
        '-fomit-frame-pointer',
        '-fipa-pure-const' if DEBUG else None,
        '-g' if DEBUG else '-g1',
        ])

    extra_compile_args_glasgow.extend([
        '-isystemexternal',
        '-DUSE_PORTABLE_SNIPPETS_BUILTIN',
        ])

    extra_compile_args_minorminer.extend([
        '-fno-rtti',
        '-DSAFE_COORDS' if SAFE_COORDS else None,
        ])

# filter out any None or empty arguments
extra_compile_args = list(filter(None, extra_compile_args))
extra_compile_args_glasgow = list(filter(None, extra_compile_args_glasgow))
extra_compile_args_minorminer = list(filter(None, extra_compile_args_minorminer))

# this is a subset of the total source files, so we can't just use glob or similar
glasgow_cc = [
    '/'.join(['external/glasgow-subgraph-solver/src', f])
    for f in [
        'cheap_all_different.cc',
        'clique.cc',
        'configuration.cc',
        'graph_traits.cc',
        'homomorphism.cc',
        'homomorphism_domain.cc',
        'homomorphism_model.cc',
        'homomorphism_searcher.cc',
        'homomorphism_traits.cc',
        'lackey.cc',
        'proof.cc',
        'restarts.cc',
        'sip_decomposer.cc',
        'svo_bitset.cc',
        'timeout.cc',
        'thread_utils.cc',
        'watches.cc',
        'formats/input_graph.cc',
        'formats/graph_file_error.cc',
    ]
]

extensions = [
    Extension(
        name="minorminer._minorminer",
        sources=["./minorminer/_minorminer.pyx"],
        include_dirs=['', './include/', './include/find_embedding'],
        language='c++',
        extra_compile_args=extra_compile_args + extra_compile_args_minorminer,
    ),
    Extension(
        name="minorminer.busclique",
        sources=["./minorminer/busclique.pyx"],
        include_dirs=['', './include/', '.include/busclique'],
        language='c++',
        extra_compile_args=extra_compile_args + extra_compile_args_minorminer,
    ),
    Extension(
        name="minorminer.subgraph",
        sources=["./minorminer/subgraph.pyx"] + glasgow_cc,
        include_dirs=['', './include', './external',
                      './external/glasgow-subgraph-solver/src'],
        library_dirs=['./include'],
        language='c++',
        extra_compile_args=extra_compile_args + extra_compile_args_glasgow,
    ),
    Extension(
        name="minorminer._extern.rpack._core",
        sources=["./minorminer/_extern/rpack/_core.pyx",
                 "./minorminer/_extern/rpack/src/rpackcore.c"],
        include_dirs=["./minorminer/_extern/rpack/include"],
        language='c',
    ),
]

setup(
    ext_modules=cythonize(extensions),
    packages=['minorminer',
              'minorminer.layout',
              'minorminer.utils',
              'minorminer._extern.rpack',
              ],
    include_package_data=True,
)
