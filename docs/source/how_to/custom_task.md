## Add a  New Task

1.  Extend `BaseTask`

```
class MyTask(BaseTask):
    def _run_task(...):
        ...
    def _compute_metrics(...):
        return [MetricResult(...)]
```

2.  Declare `required_inputs` and `required_outputs`
3.  Use `model_type`, `data.get_input()` and `data.get_output()`