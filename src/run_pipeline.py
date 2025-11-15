import argparse
import pandas as pd

from mlops_2025.preprocessing.preprocessor import Preprocessor
from mlops_2025.features.features_computer import FeatureComputer
from mlops_2025.models.model_trainer import ModelTrainer
from mlops_2025.evaluation.evaluator import Evaluator
from mlops_2025.predictor.predictor import Predictor


def build_parser():
    parser = argparse.ArgumentParser(description="Run full Titanic ML pipeline")
    parser.add_argument("--model", required=True, help="Model type: logreg, rf, xgb, etc.")
    return parser


def run_pipeline(model_name: str):
    # -------------------------------
    # Step 1: Preprocessing
    # -------------------------------
    print("Starting preprocessing...")
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")

    preprocessor = Preprocessor()
    processed_df = preprocessor.process(train, test)

    # Split back into train + test
    train_processed = processed_df.iloc[:len(train)].copy()
    test_processed = processed_df.iloc[len(train):].copy()

    print("Preprocessing complete!")

    # -------------------------------
    # Step 2: Feature Engineering
    # -------------------------------
    print("Starting feature engineering...")
    feature_engineer = FeatureComputer()
    X_train = feature_engineer.compute_features(train_processed, is_train=True).copy()
    X_test = feature_engineer.compute_features(test_processed, is_train=False).copy()
    print("Feature engineering complete!")

    # Extract target column
    y_train = train_processed["Survived"].copy()

    # Ensure target column is dropped from features
    if "Survived" in X_train.columns:
        X_train = X_train.drop(columns=["Survived"])
    if "Survived" in X_test.columns:
        X_test = X_test.drop(columns=["Survived"])

    # Save feature names for consistent prediction
    feature_names = X_train.columns.tolist()

    # -------------------------------
    # Step 3: Train Model
    # -------------------------------
    print(f"Training {model_name} model...")
    trainer = ModelTrainer(model_name=model_name)
    trainer.train(X_train, y_train)
    trainer.save_model(f"models/{model_name}.pkl")
    print(f"{model_name} model saved.")

    # -------------------------------
    # Step 4: Evaluate Model
    # -------------------------------
    print("Evaluating model...")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(trainer.model, X_train, y_train)
    print("Evaluation metrics:", metrics)

    # -------------------------------
    # Step 5: Predict
    # -------------------------------
    print("Making predictions on test data...")
    # Step 5: Predict
   
    print("Making predictions on test data...")
    predictor = Predictor()           # no arguments
    predictor.model = trainer.model   # assign the trained model
    predictions = predictor.predict(X_test)

    # Ensure output directory exists
    from pathlib import Path
    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pd.DataFrame({"Prediction": predictions}).to_csv(out_dir / "submission.csv", index=False)
    print("Pipeline finished! Predictions saved to data/predictions/submission.csv")

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args.model)
