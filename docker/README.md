# Model Implementations

This directory contains concrete model implementations. Each model should be implemented in its own subdirectory with all necessary files for containerization and execution.


## Adding New Models

1. **Create a New Directory in `docker/`**:
```bash
docker/
├── your_model/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── config.yaml
│   ├── implementation.py
│   └── assets/  # model weights, etc.
```

2. **Implement Your Model**:
```python
# docker/your_model/implementation.py

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

3. **Add Required Files**:
   - `Dockerfile`: Container setup and dependencies
   - `requirements.txt`: Python package dependencies
   - `config.yaml`: Model configuration
   - Any model-specific assets (weights, vocabularies, etc.)

4. **Update Makefile**:
   - Add a new target for building your model's Docker image

## Best Practices

- Keep all model-specific code and assets in the model's Docker directory
- Document environment requirements in Dockerfile and requirements.txt
- Use consistent naming across implementation files
- Add logging for implementation steps
- Follow existing implementation patterns (see `docker/scvi/` for an example)
