from abc import ABC, abstractmethod
from typing import Type, ClassVar
from .constants import INPUT_PATH, OUTPUT_PATH
from ..datasets.base import BaseDataset
from ..models.base import BaseModel

class BaseModelRunner(ABC):
    """Base class for model containers that handles I/O boilerplate"""
    
    model_class: ClassVar[Type['BaseModel']]
    data: BaseDataset

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define required class variables"""
        super().__init_subclass__()
        if not hasattr(cls, 'model_class'):
            raise TypeError(f"Can't instantiate {cls.__name__} without model_class class variable")
    
    @abstractmethod
    def run_model(self) -> None:
        """Implement model-specific inference logic"""
        pass
    
    def run(self):
        self.data = self.model_class.dataset_type.load(INPUT_PATH)
        
        if not self.model_class.validate_dataset(self.data):
            raise ValueError("Dataset validation failed")
        
        self.run_model()
        self.data.save(OUTPUT_PATH)

    @classmethod
    @abstractmethod
    def get_expected_dataset_type_type(cls) -> str:
        """Return the expected dataset type type for this model"""
        pass