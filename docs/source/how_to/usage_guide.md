# Usage Guide


## Example Usage

```python
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import ContainerRunner
from czbenchmarks.tasks import ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask

# Load dataset with custom configuration
dataset = load_dataset("example", config_path="custom.yaml")

# Run model
runner = ContainerRunner(
    image="czbenchmarks-scvi:latest",
    gpu=True,
)
dataset = runner.run(dataset)

# Run evaluation tasks
task = ClusteringTask(label_key="cell_type")
clustering_results = task.run(dataset)

task = EmbeddingTask(label_key="cell_type")
embedding_results = task.run(dataset)

task = MetadataLabelPredictionTask(label_key="cell_type")
prediction_results = task.run(dataset)
```
