import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from pathlib import Path


def evaluate_model(model_path, features_path, target_path=None):
    """
    Evaluate or generate predictions from a trained Logistic Regression model.
    If target_path is provided, compute accuracy.
    Otherwise, save predictions (and probabilities) to 'predictions/predicted_labels.csv'.
    """
    model = joblib.load(model_path)
    X = pd.read_csv(features_path)
    y_pred = model.predict(X)

    if target_path:
        y_true = pd.read_csv(target_path).squeeze()

        # Ensure matching lengths
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Inconsistent number of samples: "
                f"y_true={len(y_true)} vs y_pred={len(y_pred)}. "
                "Make sure you are using matching train/test data."
            )

        acc = accuracy_score(y_true, y_pred)
        print(f"[INFO] Evaluation accuracy: {acc:.4f}")
    else:
        # Save predictions and probabilities for test set
        output_dir = Path("predictions")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "predicted_labels.csv"
        probs_path = output_dir / "predicted_probabilities.csv"

        pd.DataFrame({"Prediction": y_pred}).to_csv(output_path, index=False)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            pd.DataFrame(probs, columns=[f"Prob_Class_{i}" for i in range(probs.shape[1])]).to_csv(
                probs_path, index=False
            )
            print(f"[INFO] Predictions saved to {output_path}")
            print(f"[INFO] Probabilities saved to {probs_path}")
        else:
            print(f"[INFO] Predictions saved to {output_path}")

    return y_pred


def main():
    parser = argparse.ArgumentParser(description="Evaluate or predict using trained Logistic Regression model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Logistic Regression model")
    parser.add_argument("--features", type=str, required=True, help="Path to features CSV")
    parser.add_argument("--target", type=str, default=None, help="Path to target CSV (optional)")
    args = parser.parse_args()

    evaluate_model(args.model, args.features, args.target)


if __name__ == "__main__":
    main()
