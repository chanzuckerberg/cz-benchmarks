# Use your own dataset

## Using Custom Data

To use your own data:

1. Prepare your data in a compatible format (AnnData for single-cell)
2. Create a custom configuration file (e.g., `custom.yaml`)
3. Load using the appropriate dataset type:

```python
from czbenchmarks.datasets.utils import load_dataset

dataset = load_dataset("your_dataset", config_path="custom.yaml")
```