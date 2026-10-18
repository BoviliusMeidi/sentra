from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """
    Abstact base class for computing model performance metrics.
    """
    @abstractmethod
    def evaluate(self, y_true: Any, y_pred: Any) -> Dict[str, float] :
        pass