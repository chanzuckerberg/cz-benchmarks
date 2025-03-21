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
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_markdown_builder",
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
]

viewcode_follow_imported_members = True

autoapi_keep_files = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_dirs = ["../../src/"]
# , '../../docker/geneformer',
# '../../docker/scgenept',
# '../../docker/scgpt',
# '../../docker/scvi',
# '../../docker/uce']
autoapi_type = "python"
autosummary_generate = True
autoapi_add_toctree_entry = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_type_aliases = {
    "BaseDataset": "czbenchmarks.datasets.BaseDataset",
    "Organism": "czbenchmarks.datasets.types.Organism",
}

# nitpick_ignore = [
#     ("py:class", "BaseDataset"),
#     ("py:class", "Organism"),
#     ("py:class", "czbenchmarks.metrics.types.MetricInfo.func"),
#     ("py:class", "czbenchmarks.metrics.types.MetricInfo.required_args"),
#     ("py:class", "czbenchmarks.metrics.types.MetricInfo.default_params"),
#     ("py:class", "czbenchmarks.metrics.types.MetricInfo.description"),
#     ("py:class", "czbenchmarks.metrics.types.MetricInfo.tags"),
# ]


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

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["diagram.css"]
