from abc import ABC, abstractmethod
from typing import Any, Optional
import dill
import os
import numpy as np
import pandas as pd

class BaseDataset(ABC):
    output_embedding: Optional[np.ndarray] = None
    sample_metadata: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def __init__(
        self,
        path: str,
        *args: Any,
        **kwargs: Any
    ):
        self.path = path
        self.validate()



    @abstractmethod
    def _validate(self) -> None:
        pass

    def validate(self) -> None:
        if not os.path.exists(self.path):
            raise ValueError(f"Dataset {self.path} is not valid")
        self._validate()

    def save(self, path: str) -> None:
        """
        Serialize this dataset instance to disk using dill.
        
        Args:
            path: Path where the serialized dataset should be saved
        """
        if not path.endswith(".dill"):
            path = f"{path}.dill"

        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod 
    def load(path: str) -> 'BaseDataset':
        """
        Load a serialized dataset from disk.
        
        Args:
            path: Path to the serialized dataset file
            
        Returns:
            BaseDataset: The deserialized dataset instance
        """
        if not path.endswith(".dill"):
            path = f"{path}.dill"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")

        with open(path, 'rb') as f:
            return dill.load(f)


    