# Model Validators

This directory contains validators that ensure datasets meet model requirements.

## Adding New Validators

1. **Create Validator Class**:
```python
from ..datasets.types import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator

class YourModelValidator(BaseSingleCellValidator):
    """Validation requirements for your model.

    Validates datasets for use with your model.
    Requires detailed metadata about the dataset, assay, and donor information.
    Supports multiple organisms.

    """

    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["cell_type", "batch", "donor_id"]
    required_var_keys = ["feature_name", "feature_type"]

    @property
    def inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set containing AnnData and metadata requirements
        """
        return {DataType.ANNDATA, DataType.METADATA}

    @property
    def outputs(self) -> Set[DataType]:
        """Expected model output types.

        Returns:
            Set containing embedding output type
        """
        return {DataType.EMBEDDING}
```

2. **Update __init__.py**:
   - Add your validator to `validators/__init__.py`

## Best Practices

- Document validation requirements clearly
- Use descriptive variable names
- Add logging for validation steps
- Follow existing validator patterns
- Implement comprehensive validation checks
- Support multiple organisms when possible
- Include detailed error messages

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