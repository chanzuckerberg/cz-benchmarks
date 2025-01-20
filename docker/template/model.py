"""
REQUIRED MODEL IMPLEMENTATION FILE

This file MUST:
1. Define a ModelRunner class that inherits from BaseModelRunner
2. Implement the run_model() method
3. Specify the model_class class variable

Example:
"""

from czibench.models.base import BaseModelRunner, BaseModel

class ModelRunner(BaseModelRunner):
    """
    Required container class for model implementation.
    This class will be imported and run by the base image's entrypoint.py
    """
    
    # Required: Specify which model class this container uses
    model_class = BaseModel  # Replace with your model class
    
    def run_model(self):
        """
        Required: Implement your model's inference logic here.
        self.data will contain the input dataset.
        """
        raise NotImplementedError("Model implementation required")