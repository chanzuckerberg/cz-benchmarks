# Model Validators

This directory contains validators that ensure datasets meet model requirements.

## Adding New Validators

1. **Create Validator Class**:
```python
from ..datasets.types import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator

class YourModelValidator(BaseSingleCellValidator):
    """Validation requirements for your model."""

    available_organisms = [Organism.HUMAN]
    required_obs_keys = ["cell_type", "batch"]
    required_var_keys = ["gene_symbol"]

    @property
    def inputs(self) -> Set[DataType]:
        return {DataType.ANNDATA}

    @property
    def outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    def validate_dataset(self, dataset: BaseDataset):
        """Add custom validation logic here."""
        if dataset.n_cells < 1000:
            raise ValueError("Model requires at least 1000 cells")
```

2. **Update __init__.py**:
   - Add your validator to `validators/__init__.py`

## Best Practices

- Document validation requirements clearly
- Use descriptive variable names
- Add logging for validation steps
- Follow existing validator patterns

## Example Usage

```python
from czibench.models.validators import YourModelValidator

validator = YourModelValidator()
try:
    validator.validate_dataset(dataset)
    print("Dataset validation passed!")
except ValueError as e:
    print(f"Validation failed: {e}")
```