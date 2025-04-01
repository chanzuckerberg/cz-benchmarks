# Add a new Task

## Adding New Tasks

1. Create a task class inheriting from `czbenchmarks.tasks.base.BaseTask`. For example:

```python
from czbenchmarks.tasks.base import BaseTask

class ClusteringTask(BaseTask):

    def __init__(self, label_key: str):
        self.label_key = label_key

    def validate(self, data: SingleCellDataset):
        return data.output_embedding is not None and self.label_key in data.sample_metadata.columns

    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        adata = data.adata
        adata.obsm["emb"] = data.output_embedding
        self.input_labels = data.sample_metadata[self.label_key]
        self.predicted_labels = your_label_prediction_function(...)
        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "adjusted_rand_index": adjusted_rand_index(self.input_labels, self.predicted_labels),
            "normalized_mutual_info": normalized_mutual_info(self.input_labels, self.predicted_labels),
        }
```