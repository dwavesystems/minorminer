project(
    'minorminer',
    'c', 'cpp', 'cython',
    default_options: [
        'cpp_std=c++17',
        'buildtype=release',
        'b_ndebug=if-release',  # add -DNDEBUG when building for "release", which we do by default
        'b_vscrt=mt',  # statically link the MSVC runtime (/MT), as setup.py did for the C++ extensions
        'optimization=3',
        'warning_level=1',  # equivalent to -Wall
    ],
    meson_version: '>= 1.10.0',
    # The version lives in minorminer/__init__.py; _version.py prints it.
    version: run_command(
        ['minorminer/_build/_version.py'],
        capture: true,
        check: true,
    ).stdout(),
)

py = import('python').find_installation(pure: false)

c = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# The gcc/clang flag setup.py used for the C++ extensions. MSVC doesn't support it,
# so get_supported_arguments() leaves it out there, as setup.py did.
add_project_arguments(
    cpp.get_supported_arguments('-fomit-frame-pointer'),
    language: 'cpp',
)
# setuptools also passed Python's own -fno-strict-overflow (signed integer overflow wraps
# instead of being undefined behaviour), so keep it in every build.
add_project_arguments(c.get_supported_arguments('-fno-strict-overflow'), language: 'c')
add_project_arguments(cpp.get_supported_arguments('-fno-strict-overflow'), language: 'cpp')
# setup.py's release flags, kept as they were (-Ofast includes -ffast-math). Only for
# release builds (the default, and what wheels use); other build types keep meson's flags.
if get_option('buildtype') == 'release'
    add_project_arguments(cpp.get_supported_arguments('-Ofast', '-g1'), language: 'cpp')
endif

# Whole-program optimisation for MSVC release builds, as setuptools used.
if cpp.get_id() == 'msvc' and get_option('buildtype') == 'release'
    add_project_arguments('/GL', language: ['c', 'cpp'])
    add_project_link_arguments('/LTCG', language: ['c', 'cpp'])
endif

# find_embedding (its threads parameter) and the Glasgow solver use std::thread.
threads_dep = dependency('threads')

# minorminer's own headers, used by the _minorminer and busclique extensions.
# The .pxi files include them relative to minorminer/ ("../include/...", "src/..."),
# where setup.py wrote the generated C++; meson writes it to the build directory instead.
minorminer_args = cpp.get_supported_arguments('-fno-rtti')
if get_option('buildtype') == 'debug'
    minorminer_args += '-DCPPDEBUG'
endif
if get_option('safe-coords')
    minorminer_args += '-DSAFE_COORDS'
endif
minorminer_dep = declare_dependency(
    include_directories: ['minorminer/'],
    compile_args: minorminer_args,
    dependencies: [threads_dep],
)

# The Glasgow Subgraph Solver (a git submodule), compiled into the subgraph extension.
# is_system gives -isystem on gcc/clang, as setup.py used, so warnings from its headers are
# not reported there.
glasgow_dep = declare_dependency(
    include_directories: include_directories(
        'external/',
        'external/glasgow-subgraph-solver/',
        is_system: true,
    ),
    # this is a subset of the total source files, so we can't just use all of them
    sources: files(
        'external/glasgow-subgraph-solver/gss/clique.cc',
        'external/glasgow-subgraph-solver/gss/configuration.cc',
        'external/glasgow-subgraph-solver/gss/homomorphism.cc',
        'external/glasgow-subgraph-solver/gss/innards/proof.cc',
        'external/glasgow-subgraph-solver/gss/restarts.cc',
        'external/glasgow-subgraph-solver/gss/sip_decomposer.cc',
        'external/glasgow-subgraph-solver/gss/timeout.cc',
        'external/glasgow-subgraph-solver/gss/innards/cheap_all_different.cc',
        'external/glasgow-subgraph-solver/gss/innards/graph_traits.cc',
        'external/glasgow-subgraph-solver/gss/innards/homomorphism_domain.cc',
        'external/glasgow-subgraph-solver/gss/innards/homomorphism_model.cc',
        'external/glasgow-subgraph-solver/gss/innards/homomorphism_searcher.cc',
        'external/glasgow-subgraph-solver/gss/innards/homomorphism_traits.cc',
        'external/glasgow-subgraph-solver/gss/innards/lackey.cc',
        'external/glasgow-subgraph-solver/gss/innards/svo_bitset.cc',
        'external/glasgow-subgraph-solver/gss/innards/thread_utils.cc',
        'external/glasgow-subgraph-solver/gss/innards/watches.cc',
        'external/glasgow-subgraph-solver/gss/formats/input_graph.cc',
        'external/glasgow-subgraph-solver/gss/formats/graph_file_error.cc',
    ),
    compile_args: ['-DUSE_PORTABLE_SNIPPETS_BUILTIN'],
    dependencies: [threads_dep],
)

cpp_extension_kwargs = {
    'install': true,
    'override_options': ['cython_language=cpp'],
    'subdir': 'minorminer/',
}

py.extension_module(
    '_minorminer',
    'minorminer/_minorminer.pyx',
    dependencies: [minorminer_dep],
    kwargs: cpp_extension_kwargs,
)
py.extension_module(
    'busclique',
    'minorminer/busclique.pyx',
    dependencies: [minorminer_dep],
    kwargs: cpp_extension_kwargs,
)
py.extension_module(
    'subgraph',
    'minorminer/subgraph.pyx',
    dependencies: [glasgow_dep],
    kwargs: cpp_extension_kwargs,
)

# rpack is C, not C++
py.extension_module(
    '_core',
    ['minorminer/_extern/rpack/_core.pyx', 'minorminer/_extern/rpack/src/rpackcore.c'],
    include_directories: ['minorminer/_extern/rpack/include/'],
    install: true,
    subdir: 'minorminer/_extern/rpack/',
)

install_subdir('minorminer', install_dir: py.get_install_dir(pure: false))
