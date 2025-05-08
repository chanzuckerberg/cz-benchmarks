# CZ Benchmarks

CZ Benchmarks is a package for standardized evaluation and comparison of biology-oriented machine learning models (starting with single-cell transcriptomics) across various tasks and metrics. The package provides a modular architecture for loading datasets, running containerized models, executing evaluation tasks, and computing performance metrics.

## Key Features of CZ Benchmarks:

- ✅ **Modular and Customizable**: Effortlessly integrate new models, datasets, tasks, and metrics to suit your research needs.
- 🤜 **Extensible for Innovation**: Build and expand custom benchmarks tailored to cutting-edge computational biology challenges.
- 📊 **Metrics-rich Evaluation**: Benchmark models across diverse tasks, including clustering, embedding, integration, perturbation prediction, and label prediction, using a wide array of metrics.
- 🧬 **Single-cell Native**: Designed to work seamlessly with AnnData and biological datasets, making it a perfect fit for single-cell transcriptomics research. Image modality coming soon.
- 🚀 **Scalable and Efficient**: Leverage container-based execution with GPU support for high-performance benchmarking.


## Why Choose CZ Benchmarks?

- **Reproducibility at Its Core**: Docker-based environments ensure uniformity and reproducibility across experiments.
- **Comprehensive and Insightful**: Gain deep insights into model performance with task-specific evaluations and detailed metrics.
- **User-friendly for Scientists**: Intuitive command-line interface and a well-documented Python API make it accessible for computational biologists and data scientists alike.

## Table of contents

### Getting Started
- [Quick Start Guide](docs/source/quick_start.md)

### How-To Guides
- [Add a Custom Dataset](docs/source/how_to_guides/add_custom_dataset.md)
- [Add a Custom Model](docs/source/how_to_guides/add_custom_model.md)
- [Add a New Metric](docs/source/how_to_guides/add_new_metric.md)
- [Add a New Task](docs/source/how_to_guides/add_new_task.md)
- [Interactive Mode](docs/source/how_to_guides/interactive_mode.md)
- [Visualize Results](docs/source/how_to_guides/visualize_results.md)

### Developer Guides
- [Datasets](docs/source/developer_guides/datasets.md)
- [Metrics](docs/source/developer_guides/metrics.md)
- [Models](docs/source/developer_guides/models.md)
- [Tasks](docs/source/developer_guides/tasks.md)
- [Writing Test](tests/README.md)
- [Writing Documentation](docs/README.md)


### Policies
- [Assets](docs/source/policy/assets.md)
- [Governance](docs/source/policy/governance.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

### Additional Resources
- [Changelog & Release Notes](CHANGELOG.md)
- [Roadmap](docs/source/roadmap.md)
- [Legal](docs/source/legal.md)

**Project Status: UNSTABLE**

>   ** 🚧 Under Development** - This project is under development and not yet stable. It is being actively developed, but not supported and not ready for community contribution. Things may break without notice, and it is not likely the developers will respond to requests for user support. Feedback and contributions are welcome, but user support is limited for now.


## Contributing
This project adheres to the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## Reporting Security Issues
Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.

## License Notice for Dependencies
This repository is licensed under the MIT License; however, it relies on certain third-party dependencies that are licensed under the GNU General Public License (GPL). Specifically:
- igraph (v0.11.8) is licensed under the GNU General Public License (GPL).
- leidenalg (v0.10.2) is licensed under the GNU General Public License v3 or later (GPLv3+).

These libraries are not included in this repository but must be installed separately by users. Please be aware that the GPL license terms apply to these dependencies, and certain uses of GPL-licensed code may have licensing implications for your own software.
