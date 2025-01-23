"""
REQUIRED MODEL IMPLEMENTATION FILE

This file MUST:
1. Define a model class that inherits from an existing model class
2. Implement required class variables and methods
3. Implement the run_model() method
4. Create an instance and call run() if used as main

Example implementation for a single-cell model:
"""

from czibench.models.sc import BaseYourModel


class ExampleModel(BaseYourModel):

    def run_model(self):
        """
        Required: Implement your model's inference logic here.
        Access input data via self.data.adata
        Set output embedding via self.data.output_embedding
        """
        # Add your model implementation here
        raise NotImplementedError("Model implementation required")


if __name__ == "__main__":
    ExampleModel().run()
