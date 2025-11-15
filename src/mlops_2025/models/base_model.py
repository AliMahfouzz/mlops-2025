from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class BaseModel(ABC):
    """Abstract base class for model training."""
    
    @abstractmethod
    def train(self, X, y):
        """Trains the model."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Saves the trained model to a file."""
        pass
