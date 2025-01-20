from abc import ABC, abstractmethod
from typing import Type, ClassVar
from ..datasets.base import BaseDataset

class BaseModel(ABC):
    dataset_type: ClassVar[Type[BaseDataset]]  # Type annotation for class variable
    
    def __init_subclass__(cls) -> None:
        """Validate that subclasses define required class variables"""
        super().__init_subclass__()
        if not hasattr(cls, 'dataset_type'):
            raise TypeError(f"Can't instantiate {cls.__name__} without dataset_type class variable")

    @abstractmethod
    def _validate_dataset(self, dataset: BaseDataset) -> bool:
        pass

    @classmethod
    def validate_dataset(cls, dataset: BaseDataset) -> bool:
        return isinstance(dataset, cls.dataset_type) and cls._validate_dataset(dataset)
        