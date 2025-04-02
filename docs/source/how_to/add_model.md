# How to add a new Model

## Adding a New Model

1. Create a new directory under `docker/your_model/` with:

```markdown
docker/your_model/
├── Dockerfile
├── model.py
├── config.yaml
└── requirements.txt
```

Refer to the template docker directory as a starting point (`docker/template/`)!

2. Implement your model validator in `src/czbenchmarks/models/validators/`. This should inherit from `BaseModelValidator` or `BaseSingleCellValidator` and implement:
   - Required data type specifications
   - Model-specific validation rules
   - Supported organisms and data requirements

3. Implement your model class in `model.py` (see template):

```python
from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
from czbenchmarks.models.validators.your_model import YourModelValidator

class YourModel(YourModelValidator, BaseModelImplementation):
    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        """Specify subdirectory for model weights."""
        return dataset.organism.name

    def _download_model_weights(self, dataset: BaseDataset):
        """Download model weights from your source."""
        # Implement model weight downloading logic
        pass

    def run_model(self, dataset: BaseDataset):
        """Implement your model's inference logic."""
        # Access input data via dataset.adata
        # Set output embedding via dataset.set_output()
        pass

    def parse_args(self):
        """Parse model-specific arguments."""
        pass

if __name__ == "__main__":
    YourModel().run()
```

4. Create a model configuration file (`config.yaml`)

The file `config.yaml` is used to store configuration information about the model, e.g. such as the URI for model weights and associated files. Minimally, it must contain a `_target_` entry that points to the model class based on `BaseModelImplementation` in the `model.py` file.

```yaml
_target_: model.YourModel

model_type:
  model_dir: s3://path/to/model/weights/

```

