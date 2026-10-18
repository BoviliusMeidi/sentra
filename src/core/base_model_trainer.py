from abc import ABC, abstractmethod
from typing import Any
from src.core.base_evaluator import BaseEvaluator

class BaseModelTrainer(ABC):
    """
    Abstract base class defining the Machine Learning Model Lifecycle.
    """

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    @abstractmethod
    def test_and_evaluate(self, X_test: Any, y_test: Any, evaluator: BaseEvaluator) -> dict[str, float]:
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        pass
