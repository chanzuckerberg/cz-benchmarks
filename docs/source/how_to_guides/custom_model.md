# Run with Your Own Model

## How to Add

### A New Model

1.  Create directory under `docker/your_model/`
2.  Add:
    -   `Dockerfile`, `requirements.txt`, `config.yaml`, `model.py`
3.  Create a validator class extending `BaseModelValidator`
4.  Extend `BaseModelImplementation`:
    

```
class MyModel(BaseModelValidator, BaseModelImplementation):
    def run_model(self, dataset):
        ...
```

