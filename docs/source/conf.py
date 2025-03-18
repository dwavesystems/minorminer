# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess
import os
import sys
config_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(config_directory))

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',
]
extensions += ['breathe']

autosummary_generate = True

source_suffix = '.rst'

master_doc = 'index'

# General information about the project.
project = u'minorminer'
copyright = u'2017, D-Wave Systems'
author = u'D-Wave Systems'

language = 'en'

exclude_patterns = ['_build', 'README.rst']

linkcheck_retries = 2
linkcheck_anchors = False
linkcheck_ignore = [r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    ]

pygments_style = 'sphinx'

todo_include_todos = False

# -- Breathe configuration ------------------------------------------------

# Path to the cpp xml files
breathe_projects = {"minorminer": os.path.join(
    config_directory, '../build-cpp/xml/')}

breathe_default_project = "minorminer"

breathe_default_members = ('members', )

# -- Options for HTML output ----------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

# Configuration for intersphinx.
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'dwave': ('https://docs.dwavequantum.com/en/latest/', None),
                       }

read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'
if read_the_docs_build:

    subprocess.call('cd ..; make cpp', shell=True)
