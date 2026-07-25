# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

import densemaps
import densemaps.numpy
import densemaps.torch

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'densemaps'
copyright = '2024, Robin Magnet'
author = 'Robin Magnet'
release = densemaps.__version__
version = densemaps.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              "sphinx.ext.napoleon",
              'sphinx_math_dollar',
              'sphinx.ext.mathjax',
              "myst_parser",
              "sphinx_design",
              ]

# pykeops is import-guarded in the code and only used at call time, so it need not be installed to
# build the docs. Mock it so autodoc never trips over it. (scikit-learn is a core dependency now.)
autodoc_mock_imports = ["pykeops"]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource'}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


from sphinx.ext.autodoc import between

def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect('autodoc-process-docstring', between('^.*IGNORE.*$', exclude=True))
    return app