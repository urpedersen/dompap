# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))   # Adjust the path as necessary

import dompap

project = 'dompap'
copyright = '2024, Ulf R. Pedersen'
author = 'Ulf R. Pedersen'
release = dompap.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    #'numpydoc',
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.autosectionlabel'
]

mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']