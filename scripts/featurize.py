import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def compute_features(df, mode, output_dir):
    """
    Compute features for Titanic dataset and return paths for:
    1) Feature file
    2) Target file (if present)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = output_dir / "preprocessor.pkl"

    target_col = "Survived"
    target_path = None

    # Separate target if present
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
        target_path = output_dir / f"{mode}_target.csv"
        target.to_csv(target_path, index=False)
        print(f"[INFO] Target column saved to {target_path}")

    # Define preprocessing (Titanic-specific)
    numeric_features = ["Age", "Fare"]
    categorical_features = ["Sex", "Pclass", "Embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    if mode == "train":
        # Fit preprocessor on training data
        X_processed = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"[INFO] Preprocessor fitted and saved to {preprocessor_path}")

        feature_path = output_dir / "train_features.csv"
        X_processed_df = pd.DataFrame(
            X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
        )
        X_processed_df.to_csv(feature_path, index=False)
        print(f"[INFO] Training features saved to {feature_path}")

    elif mode == "test":
        # Load pre-fitted preprocessor
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"No preprocessor found at {preprocessor_path}. Run train first.")
        preprocessor = joblib.load(preprocessor_path)
        X_processed = preprocessor.transform(df)

        feature_path = output_dir / "test_features.csv"
        X_processed_df = pd.DataFrame(
            X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
        )
        X_processed_df.to_csv(feature_path, index=False)
        print(f"[INFO] Test features saved to {feature_path}")

    else:
        raise ValueError("Mode must be 'train' or 'test'.")

    # Return both feature and target paths
    return feature_path, target_path


def main():
    parser = argparse.ArgumentParser(description="Compute Titanic dataset features")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed CSV file")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--output", type=str, required=True, help="Directory to save computed features and target")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Compute features
    feature_path, target_path = compute_features(df, args.mode, args.output)

    print(f"[INFO] Feature file path: {feature_path}")
    if target_path:
        print(f"[INFO] Target file path: {target_path}")


if __name__ == "__main__":
    main()
