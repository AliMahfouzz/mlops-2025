from abc import ABC, abstractmethod
import pandas as pd

class BaseFeatureComputer(ABC):
    """Abstract base class for feature engineering."""
    
    @abstractmethod
    def compute_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Computes features for the dataset."""
        pass
