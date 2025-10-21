import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_logistic_regression(features_path, target_path, model_output_path):
    """
    Train a Logistic Regression model using precomputed features and target.
    Saves the trained model.
    """
    # Load features and target
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()  # ensure Series

    # Initialize and train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Ensure model directory exists
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(model, model_output_path)
    print(f"[INFO] Model trained and saved to {model_output_path}")

    # Evaluate on training data
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"[INFO] Training accuracy: {acc:.4f}")

    return model



def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression on Titanic dataset")
    parser.add_argument("--train_features", type=str, required=True, help="Path to train features CSV")
    parser.add_argument("--train_target", type=str, required=True, help="Path to train target CSV")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save trained Logistic Regression model")
    args = parser.parse_args()

    # Train the model
    train_logistic_regression(args.train_features, args.train_target, args.model_output)


if __name__ == "__main__":
    main()
