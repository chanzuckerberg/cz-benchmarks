# Tasks

This directory contains all the benchmark tasks. Tasks are organized based on their scope and applicability across different modalities.

## Directory Structure

tasks/
├── init.py # Exports all available tasks
├── embedding.py # Generic embedding evaluation
├── clustering.py # Generic clustering tasks
├── label_prediction.py # Generic label prediction
└── single_cell/ # Single-cell specific tasks
├── init.py
└── custom_task.py


## Task Organization

- **Generic Tasks**: Tasks that can be applied across multiple modalities (e.g., embedding evaluation, clustering, label prediction) should be placed directly in the `tasks/` directory.
  - Each task should be in its own file (e.g., `embedding.py`, `clustering.py`)
  - Task names should clearly indicate their purpose
  - Generic tasks should avoid modality-specific dependencies

- **Specialized Tasks**: Tasks specific to a particular modality should be placed in their own subdirectory:
  - `single_cell/` for single-cell specific tasks
  - `imaging/` for imaging tasks
  - etc.

## Adding New Tasks

1. **Choose the Right Location**:
   - If your task can work across modalities → add to `tasks/`
   - If specific to one modality → add to appropriate subdirectory (create if needed)

2. **Create Task File**:
   ```python
   from ..base import BaseTask

   class YourNewTask(BaseTask):
       def __init__(self, **kwargs):
           # Initialize your task parameters
           pass

       def _run_task(self, data):
           # Implement your task logic
           pass

       def _compute_metrics(self):
           # Return task-specific metrics
           return {"metric_name": value}
   ```

3. **Update __init__.py**:
   - Add your task to the appropriate `__init__.py`
   - For generic tasks, add to `tasks/__init__.py`
   - For specialized tasks, add to the modality-specific `__init__.py`

4. **Documentation**:
   - Add docstrings to your task class and methods
   - Update this README if adding a new modality directory

## Best Practices

- Keep tasks focused and single-purpose
- Document input/output requirements clearly
- Follow existing task patterns for consistency
- Use type hints for better code clarity
- Add logging for important task steps

## Example Task Usage

```python
from czibench.tasks import EmbeddingTask
# Initialize task
task = EmbeddingTask(label_key="cell_type")
# Run task
results = task.run(data)
```