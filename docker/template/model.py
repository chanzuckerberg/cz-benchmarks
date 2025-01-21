"""
REQUIRED MODEL IMPLEMENTATION FILE

This file MUST:
1. Define a model class that inherits from SingleCellModel (for single-cell data) 
   or BaseModel (for other modalities)
2. Implement required class variables and methods
3. Implement the run_model() method
4. Create an instance and call run() if used as main

Example implementation for a single-cell model:
"""

from czibench.models.sc import SingleCellModel
from czibench.datasets.types import Organism
from czibench.datasets.sc import SingleCellDataset

class ExampleModel(SingleCellModel):
    # Required: Specify which organisms this model supports
    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    
    # Required: Specify which metadata columns are needed for batching
    required_obs_keys = ['dataset_id', 'donor_id']  # Add required columns

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset) -> bool:
        """
        Required: Implement validation logic specific to your model
        Args:
            dataset: The input SingleCellDataset to validate
        Returns:
            bool: True if dataset meets model requirements, False otherwise
        """
        # Check if all required batch keys are present
        missing_keys = [key for key in cls.required_obs_keys 
                       if key not in dataset.adata.obs.columns]
        
        if missing_keys:
            return False
            
        return True
    
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