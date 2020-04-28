from setuptools import setup, extension
from Cython.Build import cythonize


extensions = cythonize([extension.Extension(
    name="busclique.clique_experiment",
    sources=["busclique/clique_experiment.pyx"],
    include_dirs=['', './include/'],
    language='c++',
    extra_compile_args = ['-std=c++11', '-DEBUG', '-g', '-O0']#, '-fno-rtti', '-Ofast']
)])

setup(
    name="busclique",
    ext_modules=extensions,
    packages=['busclique'],
)
