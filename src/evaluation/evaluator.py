from .base_evaluator import BaseEvaluator
from sklearn.metrics import accuracy_score

class Evaluator(BaseEvaluator):
    def evaluate(self, model, X, y):
        preds = model.predict(X)
        return accuracy_score(y, preds)
