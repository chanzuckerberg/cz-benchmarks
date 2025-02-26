from czibench.tasks.sc import MetadataLabelPredictionTask
from czibench.datasets.utils import load_dataset
from czibench.datasets.types import DataType

dataset = load_dataset("example", config_path="custom.yaml")

# Here we set the embedding to be the raw counts
dataset.set_output(DataType.EMBEDDING, dataset.adata.X.toarray())

task = MetadataLabelPredictionTask(label_key="cell_type")

# This now contains the linear baseline prediction results
linear_baseline_prediction_results = task.run(dataset)
