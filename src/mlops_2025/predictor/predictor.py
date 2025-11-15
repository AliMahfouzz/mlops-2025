import pandas as pd
import pickle
from pathlib import Path
from .base_predictor import BasePredictor

class Predictor(BasePredictor):
    """Concrete implementation of prediction logic."""

    def __init__(self, model_path: str = None):
        """
        Optionally load a model at creation.
        """
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained model from file."""
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, df: pd.DataFrame):
        """Use the trained model to generate predictions."""
        if self.model is None:
            raise ValueError("Model is not loaded. Use load_model() first or pass model_path in constructor.")
        return self.model.predict(df)

    def predict_from_files(self, model_path: str, input_csv: str, output_csv: str):
        """
        File-based prediction method.
        """
        # Ensure output directory exists
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

        # Load model
        self.load_model(model_path)

        # Load input data
        df = pd.read_csv(input_csv)

        # Generate predictions
        preds = self.predict(df)

        # Save results
        output = pd.DataFrame({"Prediction": preds})
        output.to_csv(output_csv, index=False)

        print(f"Predictions saved to {output_csv}")

        return preds