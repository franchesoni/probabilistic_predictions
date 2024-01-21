
from base_methods import BaseDeterministicPredictor, BaseProbabilisticPredictor, MLP, fit_deterministic_model_torch, predict_deterministic_model_torch


class SimplestDeterministicPredictor(BaseDeterministicPredictor):
    def __init__(self):
        self.model = MLP(1)

    def fit(self, X, y):
        self.model = fit_deterministic_model_torch(self.model, X, y)
        return self.model

    def predict(self, X):
        return predict_deterministic_model_torch(self.model, X).numpy()

methods = [
    SimplestDeterministicPredictor(),
]