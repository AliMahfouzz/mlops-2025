import pandas as pd

class Splitter:
    def __init__(self, split_index: int):
        self.split_index = split_index

    def split(self, dataFrame: pd.DataFrame):
        if not 0 < self.split_index < len(dataFrame):
            raise ValueError("split_index must be between 0 and the number of rows in the DataFrame")

        train_df = dataFrame.iloc[:self.split_index].copy()
        test_df = dataFrame.iloc[self.split_index:].copy()

        return train_df, test_df
