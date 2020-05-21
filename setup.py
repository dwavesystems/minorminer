from setuptools import setup, extension
from Cython.Build import cythonize


extensions = cythonize([extension.Extension(
    name="busclique.busclique",
    sources=["busclique/busclique.pyx"],
    include_dirs=['', './include/'],
    language='c++',
#    extra_compile_args = ['-std=c++11', '-fno-rtti', '-DEBUG', '-g', '-O0']
    extra_compile_args = ['-std=c++11', '-fno-rtti', '-Ofast']
)])

setup(
    name="busclique",
    ext_modules=extensions,
    packages=['busclique'],
)
