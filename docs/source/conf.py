# Configuration file for the Sphinx documentation builder.
import os
import sys

import toml

sys.path.insert(0, os.path.abspath("../../src"))
with open("../../pyproject.toml", "r") as f:
    config = toml.load(f)


latest_version = config["project"]["version"]


project = "cz-benchmarks"
copyright = "2025, Chan Zuckerberg Initiative"
author = "Chan Zuckerberg Initiative"
release = str(latest_version)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages",
    "sphinx_markdown_builder",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "autoapi.extension",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]


viewcode_follow_imported_members = True
autosummary_generate = True

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_dirs = ["../../src/"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_keep_files = True


napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_fence_as_directive = ["mermaid"]

mermaid_d3_zoom = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "linkify",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# autodoc_type_aliases = {
#     "BaseDataset": "czbenchmarks.datasets.BaseDataset",
#     "Organism": "czbenchmarks.datasets.types.Organism",
# }


inheritance_graph_attrs = dict(
    rankdir="LR", size='"18.0, 28.0 "', fontsize=16, ratio="expand", dpi=96
)

inheritance_node_attrs = dict(
    shape="box",
    fontsize=16,
    height=1,
    color="lightblue",
    style="filled",
    fontcolor="black",
)

inheritance_edge_attrs = dict(color="gray", arrowsize=1.2, style="solid")

html_css_files = ["custom.css"]

nbsphinx_allow_errors = True
