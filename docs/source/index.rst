.. cz-benchmarks documentation master file, created by
   sphinx-quickstart on Wed Mar 19 11:27:15 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CZ Benchmarks Documentation
===========================

CZ Benchmarks is a reproducible benchmarking package designed for standardized evaluation and comparison of biology-oriented machine learning models (starting with single-cell transcriptomics) across various tasks and metrics. It supports plug-and-play benchmarking of models and tasks using Docker containers, with support for custom models, datasets, and evaluation metrics.


Key Features of CZ Benchmarks:
------------------------------

- ✅ **Modular and Customizable**: Effortlessly integrate new models, datasets, tasks, and metrics to suit your research needs.
- 🤜 **Extensible for Innovation**: Build and expand custom benchmarks tailored to cutting-edge computational biology challenges.
- 📊 **Metrics-rich Evaluation**: Benchmark models across diverse tasks, including clustering, embedding, integration, perturbation prediction, and label prediction, using a wide array of metrics.
- 🧬 **Single-cell Native**: Designed to work seamlessly with AnnData and biological datasets, making it a perfect fit for single-cell transcriptomics research. Image modality coming soon.
- 🚀 **Scalable and Efficient**: Leverage container-based execution with GPU support for high-performance benchmarking.


Why Choose CZ Benchmarks?
-------------------------

- **Reproducibility at Its Core**: Docker-based environments ensure uniformity and reproducibility across experiments.
- **Comprehensive and Insightful**: Gain deep insights into model performance with task-specific evaluations and detailed metrics.
- **User-friendly for Scientists**: Intuitive command-line interface and a well-documented Python API make it accessible for computational biologists and data scientists alike.

**Project Status: UNSTABLE**

.. warning::

   ** 🚧 Under Development** - This project is under development and not yet stable. It is being actively developed, but not supported and not ready for community contribution. Things may break without notice, and it is not likely the developers will respond to requests for user support. Feedback and contributions are welcome, but user support is limited for now.


**Project Roadmap:**

For upcoming features and plans, see the :doc:`Project Roadmap <roadmap>`.


.. toctree::
   :maxdepth: 1

   quick_start
   how_to_guides/index
   developer_guides/index
   api_reference
   policy/index
   roadmap
   legal
   changelog_release_notes


