# Base Model Implementations

This directory contains the **base classes** for models implementations.

**Concrete model implementations should be added to the `docker/` directory, not here.**

## Base Model Implementation

The `BaseModelImplementation` class defines the interface that all model implementations must follow.

### Required Methods

1. **Model Weights Management**
   ```python
   @abstractmethod
   def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
       """Get subdirectory for model variant weights."""
   
   @abstractmethod
   def _download_model_weights(self, dataset: BaseDataset):
       """Download model weights for the given dataset."""
   ```

2. **Model Execution**
   ```python
   @abstractmethod
   def run_model(self, dataset: BaseDataset) -> None:
       """Run the model on the given dataset."""
   
   @abstractmethod
   def parse_args(self):
       """Parse command line arguments."""
   ```

### Properties

1. **Dataset Management**
   ```python
   datasets: List[BaseDataset]  # List of datasets to process - automatically populated by the inference boilerplate
   model_weights_dir: str       # Directory containing model weights - automatically set by model caching boilerplate
   ```

## Best Practices

1. **Model Weights**
   - Implement proper weight downloading
   - Handle weight versioning
   - Validate downloaded weights

2. **Error Handling**
   - Validate inputs before processing
   - Provide clear error messages
   - Handle edge cases gracefully

3. **Performance**
   - Implement efficient data loading
   - Use appropriate batch sizes
   - Monitor memory usage

## Example Usage

```python
from .base_model_implementation import BaseModelImplementation
from ..datasets.base import BaseDataset

class YourModelImplementation(BaseModelImplementation):
    def __init__(self):
        self.datasets = []
        self.model_weights_dir = "path/to/weights"
    
    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        return f"your_model/{dataset.name}"
    
    def _download_model_weights(self, dataset: BaseDataset):
        # Implement weight downloading
        pass
    
    def run_model(self, dataset: BaseDataset) -> None:
        # Implement model execution
        pass
    
    def parse_args(self):
        # Implement argument parsing
        pass
```
