from abc import ABC, abstractmethod
import pandas as pd

class BasePredictor(ABC):
    """Abstract base class for prediction step."""

    @abstractmethod
    def predict(self, model, df: pd.DataFrame):
        """
        Takes a trained model and an input DataFrame.
        Returns predictions.
        """
        pass
