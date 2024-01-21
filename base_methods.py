from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn


class BaseProbabilisticPredictor(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_distribution(self, X):
        pass


class BaseDeterministicPredictor(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def predict_distribution(self, X, support_start, support_end):
        assert support_start < support_end
        prediction = self.predict(X)
        assert prediction >= support_start and prediction <= support_end
        bins = np.array([support_start, prediction, support_end])
        probs = np.array([0.5, 0.5])
        return bins, probs

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, act_layer=nn.ReLU, dropout=0., bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act_layer = act_layer(inplace=True) if act_layer else nn.Identity()
        self.norm_layer = norm_layer(num_features=output_dim) if norm_layer else nn.Identity()
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, x):
        z = self.dropout(x)
        z = self.linear(z)
        if z.shape == x.shape:
            z += x
        z = self.norm_layer(z)
        z = self.act_layer(z)
        return z


class MLP(nn.Module):
    def __init__(self, output_dim, input_dim=2, hidden_dim=32, n_hidden=2):
        super().__init__()
        assert n_hidden >= 2
        self.initial_layer = ResidualBlock(input_dim, hidden_dim)
        self.final_layer = ResidualBlock(hidden_dim, output_dim, act_layer=None, norm_layer=None)
        hidden = []
        for i in range(1, n_hidden-1):
            hidden.append(ResidualBlock(hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(hidden)

    def forward(self, x):
        z = self.initial_layer(x)
        for layer in self.hidden:
            z = layer(z)
        z = self.final_layer(z)
        return z



def fit_deterministic_model_torch(
    model,
    X,
    y,
    loss_fn=nn.functional.l1_loss,
    batch_size=16384,
    lr=1e-2,
    n_epochs=24,
    optim="adamw",
    verbose=False,
):
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=lr)
        if optim == "adamw"
        else torch.optim.SGD(model.parameters(), lr=lr)
    )
    model.train()
    for i in range(n_epochs):  # epochs
        for j in range(0, X.shape[0], batch_size):  # batches
            X_batch = X[j : j + batch_size]
            y_batch = y[j : j + batch_size]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Epoch {i+1}/{n_epochs}, loss={loss.item():.4f}")
    return model


def predict_deterministic_model_torch(model, X, batch_size=16384):
    model.eval()
    with torch.no_grad():
        y_pred = torch.zeros(X.shape[0])
        for j in range(0, X.shape[0], batch_size):
            X_batch = X[j : j + batch_size]
            y_pred[j : j + batch_size] = model(X_batch).squeeze()
    return y_pred
