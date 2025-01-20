import sys
from importlib import import_module
from czibench.runner.base import BaseModelRunner

def validate_model_implementation():
    try:
        model_module = import_module('model')
        if not hasattr(model_module, 'ModelRunner'):
            raise ImportError("model.py must define ModelRunner class")
        
        if not issubclass(model_module.ModelRunner, BaseModelRunner):
            raise TypeError("ModelRunner must inherit from BaseModelRunner")
            
    except ImportError as e:
        raise ImportError("Required file model.py not found or invalid") from e

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--data_type":
        # Print metadata as JSON for container inspection
        from model import ModelRunner
        print(ModelRunner.get_expected_dataset_type_type())
        sys.exit(0)
        
    validate_model_implementation()
    from model import ModelRunner
    ModelRunner().run()