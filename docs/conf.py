# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from os import path

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../ziopcpy"))

sys.path.insert(0, target_dir)
# check if dir is correct and list all files
print(current_dir)
print(target_dir)
print(os.listdir())

# Check if dependencies are there
try:
    import pandas
    print "pandas: %s, %s" % (pandas.__version__, pandas.__file__)
except ImportError:
    print "no pandas"
try:
    import numpy
    print "numpy: %s, %s" % (numpy.__version__, numpy.__file__)
except ImportError:
    print "no numpy"
try:
    import scipy
    print "scipy: %s, %s" % (scipy.__version__, scipy.__file__)
except ImportError:
    print "no scipy"
# -- Project information -----------------------------------------------------

project = 'ZiopcPy'
copyright = '2020, Nguyen Huynh'
author = 'Nguyen Huynh'
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.ifconfig', 'sphinx.ext.viewcode',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.autosummary',
              'numpydoc'  # numpydoc or sphinx.ext.napoleon, but not both
              ]
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
