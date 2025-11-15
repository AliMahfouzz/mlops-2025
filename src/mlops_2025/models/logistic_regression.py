from .base_model import BaseModel
import pickle
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
