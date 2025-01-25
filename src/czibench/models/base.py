from abc import ABC, abstractmethod
from typing import Type, ClassVar
from ..datasets.base import BaseDataset
from ..constants import INPUT_DATA_PATH_DOCKER, OUTPUT_DATA_PATH_DOCKER

class BaseModel(ABC):
    dataset_type: ClassVar[Type[BaseDataset]]  # Type annotation for class variable
    data: BaseDataset

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define required class variables"""
        super().__init_subclass__()
        if not hasattr(cls, 'dataset_type'):
            raise TypeError(f"Can't instantiate {cls.__name__} without dataset_type class variable")

    @abstractmethod
    def _validate_dataset(self, dataset: BaseDataset) -> bool:
        pass

    @classmethod
    def validate_dataset(cls, dataset: BaseDataset):
        if not isinstance(dataset, cls.dataset_type):
            raise ValueError(f"Dataset type mismatch: expected {cls.dataset_type.__name__}, got {type(dataset).__name__}")
        
        cls._validate_dataset(dataset)
        

    @abstractmethod
    def run_model(self) -> None:
        """Implement model-specific inference logic"""
        pass
    
    def run(self):
        self.data = self.dataset_type.load(INPUT_DATA_PATH_DOCKER)
        
        self.validate_dataset(self.data)
        
        self.run_model()
        self.data.save(OUTPUT_DATA_PATH_DOCKER)
        