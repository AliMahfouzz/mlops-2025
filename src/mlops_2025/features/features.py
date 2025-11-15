from .base_features import BaseFeatureComputer
import pandas as pd

class FeatureComputer(BaseFeatureComputer):
    def compute_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        # Create FamilySize feature
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

        # Base columns used for both train and test
        base_columns = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

        # Add 'Survived' only if it's in the dataset (for training)
        if is_train and "Survived" in df.columns:
            columns = base_columns + ["Survived"]
        else:
            columns = base_columns

        # Keep only the selected columns
        df = df[columns]

        # Convert categorical variables into dummy/indicator variables
        df = pd.get_dummies(df, drop_first=True)

        return df
