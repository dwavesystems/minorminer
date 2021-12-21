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

extra_compile_args = {
    'msvc': ['/std:c++latest', '/MT', '/EHsc', '/O2' ],
    'unix': ['-std=c++11', '-Wall', '-Wno-format-security', '-Ofast', '-fomit-frame-pointer', '-DNDEBUG', '-fno-rtti'],
}

extra_link_args = {
    'msvc': [],
    'unix': ['-std=c++11'],
}


if '--debug' in sys.argv or '-g' in sys.argv or 'CPPDEBUG' in os.environ:
    extra_compile_args['msvc'].append('/DCPPDEBUG')
    extra_compile_args['unix'] = ['-std=c++1y', '-Wall',# '-O0',
                                  '-g', '-fipa-pure-const', '-DCPPDEBUG']

if 'SAFE_COORDS' in os.environ:
    extra_compile_args['msvc'].append('/DSAFE_COORDS')
    extra_compile_args['unix'].append('-DSAFE_COORDS')

class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type

        compile_args = extra_compile_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args = compile_args

        link_args = extra_link_args[compiler]
        for ext in self.extensions:
            ext.extra_link_args = link_args

        build_ext.build_extensions(self)


class Extension(extension.Extension, object):
    pass


ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [
    Extension(
        name="minorminer._minorminer",
        sources=["./minorminer/_minorminer" + ext],
        include_dirs=['', './include/', './include/find_embedding'],
        language='c++',
    ),
    Extension(
        name="minorminer.busclique",
        sources=["./minorminer/busclique" + ext],
        include_dirs=['', './include/', '.include/busclique'],
        language='c++',
    ),
]

if USE_CYTHON:
    extensions = cythonize(extensions)

os.environ["MACOSX_DEPLOYMENT_TARGET"] = platform.mac_ver()[0]

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]

python_requires = '>=3.6'
install_requires = [
    "scipy", "networkx", "dwave-networkx>=0.8.10", "numpy", "fasteners", "homebase", "rectangle-packer>=2.0.1"
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
