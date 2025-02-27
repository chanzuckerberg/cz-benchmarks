# Model Implementations

This directory contains concrete implementations of models.

## Adding New Models

1. **Create Model Implementation**:
```python
from ...models.base_model_implementation import BaseModelImplementation
from ...models.validators import YourModelValidator

class YourModelImplementation(BaseModelImplementation, YourModelValidator):
    """Implementation of your model."""

    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        """Specify where model weights are stored."""
        return "your_model/weights"

    def _download_model_weights(self, dataset: BaseDataset):
        """Download model weights from storage."""
        # Download weights from S3, etc.
        pass

    def run_model(self, dataset: BaseDataset):
        """Run model inference."""
        embeddings = self.model.encode(dataset.adata)
        dataset.set_output(DataType.EMBEDDING, embeddings)

    def parse_args(self):
        """Parse model-specific arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=32)
        return parser.parse_args()
```

2. **Update __init__.py**:
   - Add your implementation to `implementations/__init__.py`

## Best Practices

- Document implementation requirements clearly
- Use descriptive variable names
- Add logging for implementation steps
- Follow existing imlpementation patterns

## Example Usage

```python
from czibench.models.implementations import YourModelImplementation

model = YourModelImplementation()
model.run()  # Handles dataset loading, validation, and inference
```