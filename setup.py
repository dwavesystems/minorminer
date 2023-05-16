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

_PY2 = sys.version_info.major == 2

# Change directories so this works when called from other locations. Useful in build systems that build from source.
setup_folder_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_folder_loc)

# Add __version__, __author__, __authoremail__, __description__ to this namespace
path_to_package_info = os.path.join('.', 'minorminer', 'package_info.py')
if _PY2:
    execfile(path_to_package_info)
else:
    exec(open(path_to_package_info).read())

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
            ext.extra_compile_args = extra_compile_args[arg_key][compiler]

        link_args = extra_link_args[compiler]
        for ext in self.extensions:
            ext.extra_link_args = link_args

        build_ext.build_extensions(self)
        build_ext.build_extensions(self)


class Extension(extension.Extension, object):
    pass


ext = '.pyx' if USE_CYTHON else '.cpp'

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
]

if USE_CYTHON:
    extensions = cythonize(extensions)

os.environ["MACOSX_DEPLOYMENT_TARGET"] = platform.mac_ver()[0]

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

python_requires = '>=3.7'
install_requires = [
    "dwave-networkx>=0.8.10",
    "fasteners>=0.15",
    "homebase>=1.0.1",
    "networkx>=2.4",
    "numpy>=1.21.6",
    "rectangle-packer>=2.0.1",
    "scipy>=1.7.3",
]

setup(
    name="minorminer",
    description=__description__,
    long_description="minorminer is a tool for finding graph minors, developed to embed Ising problems onto quantum annealers (QA). Where it can be used to find minors in arbitrary graphs, it is particularly geared towards the state of the art in QA: problem graphs of a few to a few hundred variables, and hardware graphs of a few thousand qubits.",
    author=__author__,
    author_email=__authoremail__,
    url="https://github.com/dwavesystems/minorminer",
    version=__version__,
    license="Apache 2.0",
    ext_modules=extensions,
    packages=['minorminer',
              'minorminer.layout',
              'minorminer.utils',
              ],
    classifiers=classifiers,
    python_requires=python_requires,
    install_requires=install_requires,
    cmdclass={'build_ext': build_ext_compiler_check}
)
