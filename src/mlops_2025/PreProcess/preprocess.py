from .basePreprocessor import BasePreprocessor
import pandas as pd

class Preprocessor(BasePreprocessor):
    def process(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and dropping unnecessary columns."""
        # Drop Cabin column due to numerous null values
        train.drop(columns=["Cabin"], inplace=True)
        test.drop(columns=["Cabin"], inplace=True)

        # Fill missing values
        train["Embarked"].fillna("S", inplace=True)
        test["Fare"].fillna(test["Fare"].mean(), inplace=True)

        # Create unified dataframe for easier manipulation
        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        # Fill missing Age values using group median
        df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )

        return df