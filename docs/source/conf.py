# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os, sys
sys.path.insert(0, os.path.abspath('../..'))
# Fix: export PYTHONPATH="${PYTHONPATH}:/home/$USER/PycharmProjects/dompap"


import dompap

project = 'dompap'
# copyright = '2024, Ulf R. Pedersen'
author = 'Ulf R. Pedersen'
release = dompap.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    #'numpydoc',
    'myst_nb',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    # 'sphinx.ext.autosectionlabel'
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}
myst_enable_extensions = ["dollarmath", "amsmath"]

# Add logo
# dompap = '_static/dompap_logo_80x80.png'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']



html_theme_options = {
  "show_toc_level": 4,
  "nosidebar": True,
}
