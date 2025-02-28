# Models

This directory contains the base classes and implementations for model validation and execution.

## Directory Structure

```
models/
├── __init__.py
├── README.md
├── implementations/                  # Model implementations
│   ├── __init__.py
│   ├── base_model_implementation.py  # Base implementation class
│   └── README.md
└── validators/                       # Model validators
    ├── __init__.py
    ├── base_model_validator.py       # Base validator class
    ├── base_single_cell_model_validator.py
    ├── <model-specific-validator>.py
    └── README.md
```

## Overview
- `implementations/`: Contains model-specific implementations
- `validators/`: Contains model-specific validation rules

See subdirectory READMEs for details on adding new models and validators.