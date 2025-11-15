from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
import pandas as pd

class BaseEvaluator(ABC):
    """Abstract base class for model evaluation."""
    
    @abstractmethod
    def evaluate(self, model, X, y):
        """Evaluates the model on provided data."""
        pass
