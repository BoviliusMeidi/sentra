from abc import ABC, abstractmethod
from typing import Any

class BaseDataProcessor(ABC):
    """
    Abstact base class for manipulating and processing raw spatial data.
    """
    @abstractmethod
    def process(self, *args, **kwargs) -> Any :
        pass