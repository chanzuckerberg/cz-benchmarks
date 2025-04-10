# Run with Your Own Dataset

## Using Custom Data

1.  Prepare your data in `.h5ad` (AnnData) format
2.  Ensure required metadata columns are present
3.  Create `custom.yaml` config file:
    

```
datasets:
  my_dataset:
    _target_: czbenchmarks.datasets.SingleCellDataset
    path: ~/my_data.h5ad
    organism: HUMAN
```

4.  Load in Python:
    

```
from czbenchmarks.datasets.utils import load_dataset

dataset = load_dataset("my_dataset", config_path="custom.yaml")
```


