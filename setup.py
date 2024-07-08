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

from setuptools import setup, extension
from setuptools.command.build_ext import build_ext
import sys
import os
import platform
cwd = os.path.abspath(os.path.dirname(__file__))

if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
    try:
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError:
        USE_CYTHON = False
else:
    USE_CYTHON = False

# Change directories so this works when called from other locations. Useful in build systems that build from source.
setup_folder_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_folder_loc)

base_compile_args = {
    'msvc': ['/std:c++latest', '/MT', '/EHsc', '/O2'],
    'unix': ['-std=c++17', '-Wall', '-Wno-format-security', '-Ofast', '-fomit-frame-pointer',
             '-DNDEBUG'],
}

mm_compile_args = {
    'msvc': base_compile_args['msvc'] + [],
    'unix': base_compile_args['unix'] + ['-fno-rtti']
}

if '--debug' in sys.argv or '-g' in sys.argv or 'CPPDEBUG' in os.environ:
    mm_compile_args['msvc'].append('/DCPPDEBUG')
    base_compile_args['unix'] = ['-std=c++17', '-Wall', '-O0', '-g', '-fipa-pure-const']


glasgow_compile_args = {
    'msvc': base_compile_args['msvc'] + ['/external:W4', '/external:I external', '/DUSE_PORTABLE_SNIPPETS_BUILTIN'],
    'unix': base_compile_args['unix'] + ['-isystemexternal', '-DUSE_PORTABLE_SNIPPETS_BUILTIN']
}

if 'SAFE_COORDS' in os.environ:
    mm_compile_args['msvc'].append('/DSAFE_COORDS')
    mm_compile_args['unix'].append('-DSAFE_COORDS')

extra_compile_args = {
    'mm': mm_compile_args,
    'glasgow': glasgow_compile_args,
}

extra_link_args = {
    'msvc': [],
    'unix': ['-std=c++17'],
}

class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type

        for ext in self.extensions:
            arg_key = ext.extra_compile_args
            if arg_key:
                ext.extra_compile_args = extra_compile_args[arg_key][compiler]

        link_args = extra_link_args[compiler]
        for ext in self.extensions:
            ext.extra_link_args = link_args

        build_ext.build_extensions(self)
        build_ext.build_extensions(self)


class Extension(extension.Extension, object):
    pass


ext = '.pyx' if USE_CYTHON else '.cpp'
ext_c = '.pyx' if USE_CYTHON else '.c'

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
        sources=["./minorminer/_minorminer" + ext],
        include_dirs=['', './include/', './include/find_embedding'],
        language='c++',
        extra_compile_args = 'mm',
    ),
    Extension(
        name="minorminer.busclique",
        sources=["./minorminer/busclique" + ext],
        include_dirs=['', './include/', '.include/busclique'],
        language='c++',
        extra_compile_args = 'mm',
    ),
    Extension(
        name="minorminer.subgraph",
        sources=["./minorminer/subgraph" + ext] + glasgow_cc,
        include_dirs=['', './include', './external', './external/glasgow-subgraph-solver/src'],
        library_dirs=['./include'],
        language='c++',
        extra_compile_args = 'glasgow',
    ),
    Extension(
        name="minorminer._extern.rpack._core",
        sources=["./minorminer/_extern/rpack/_core" + ext_c, "./minorminer/_extern/rpack/src/rpackcore.c"],
        include_dirs=["./minorminer/_extern/rpack/include"],
        language='c',
    ),
]

if USE_CYTHON:
    extensions = cythonize(extensions)

setup(
    ext_modules=extensions,
    packages=['minorminer',
              'minorminer.layout',
              'minorminer.utils',
              'minorminer._extern.rpack',
              ],
    cmdclass={'build_ext': build_ext_compiler_check},
    package_data={"minorminer._extern.rpack._core": ["_core.pyx"]},
    include_package_data=True,
)
