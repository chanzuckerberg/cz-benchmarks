# Installing cz-benchmarks

## Installation Instructions

### From Source

```bash
git clone https://github.com/chanzuckerberg/cz-benchmarks.git
cd cz-benchmarks
pip install .
```

### macOS Development Setup

When developing on macOS, first install hnswlib from conda-forge:

```bash
conda install -c conda-forge hnswlib
```

Then proceed with the regular installation. For development, include the dev dependencies:
```bash
pip install -e ".[dev]"
```
