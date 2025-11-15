import os
import pandas as pd

class DataLoader:
    def __init__(self, withdirectory: bool = True):
        self.withdirectory = withdirectory

    def load(self, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        return pd.read_csv(filepath)

    def save(self, filepath: str, dataFrame: pd.DataFrame):
        directory = os.path.dirname(filepath)

        if self.withdirectory and directory:
            os.makedirs(directory, exist_ok=True)

        dataFrame.to_csv(filepath, index=False)
        print(f"DataFrame saved successfully to {filepath}")
