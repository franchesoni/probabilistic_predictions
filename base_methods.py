from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import tqdm


class BaseProbabilisticMethod(ABC):
    @abstractmethod
    def get_mlp_output_dim(self):
        """Get the output dimension of the MLP."""
        pass

    @abstractmethod
    def predict_cdf(self, dist_param_prediction, point_to_evaluate):
        """Get a cdf depending on the predicted parameters and evaluate it at one point."""
        pass

    @abstractmethod
    def get_cdf_func(self, dist_param_prediction):
        """Get a cdf depending on the predicted parameters."""
        pass

    @abstractmethod
    def loss_fn(self, y_pred, y_target):
        """Loss to be minimized via gradient descent."""
        pass


class ResidualBlock(nn.Module):
    """Residual block with normalization and activation."""
    def __init__(
        self,
        input_dim,
        output_dim,
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.ReLU,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act_layer = act_layer(inplace=True) if act_layer else nn.Identity()
        self.norm_layer = (
            norm_layer(num_features=output_dim) if norm_layer else nn.Identity()
        )
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
    """MLP with residual blocks."""
    def __init__(self, output_dim, input_dim=2, hidden_dim=64, n_hidden=4):
        super().__init__()
        assert n_hidden >= 2
        self.output_dim = output_dim
        self.initial_layer = ResidualBlock(input_dim, hidden_dim)
        self.final_layer = ResidualBlock(
            hidden_dim, output_dim, act_layer=None, norm_layer=None
        )
        hidden = []
        for i in range(1, n_hidden - 1):
            hidden.append(ResidualBlock(hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(hidden)

    def forward(self, x):
        z = self.initial_layer(x)
        for layer in self.hidden:
            z = layer(z)
        z = self.final_layer(z)
        return z


def fit_torch(
    model,
    X,
    y,
    loss_fn=nn.functional.l1_loss,
    batch_size=16384,
    lr=1e-1,
    n_epochs=24,
    optim="sgd",
    verbose=True,
    extra_metrics={},
    extra_metrics_every=1,
):
    """Fit a torch model using gradient descent."""
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=lr)
        if optim == "adamw"
        else torch.optim.SGD(model.parameters(), lr=lr)
        if optim == "sgd"
        else NotImplementedError(f"optimizer {optim} not implemented")
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=X.shape[0] // batch_size + 1,
        epochs=n_epochs,
    )
    model.train()
    for i in range(n_epochs):  # epochs
        for j in tqdm.tqdm(range(0, X.shape[0], batch_size)):  # batches
            X_batch = X[j : j + batch_size]
            y_batch = y[j : j + batch_size]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if i % extra_metrics_every == 0:
            with torch.no_grad():
                extras = {}
                for name, metric in extra_metrics.items():
                    extras[name] = metric(y_pred[:100], y_batch[:100])
        if verbose:
            print(
                f"Epoch {i+1}/{n_epochs}, loss={loss.item():.4f}, extras={extras}",
                end="\n",
            )
    return model


def predict_torch(model, X, batch_size=16384):
    """Predict using a torch model."""
    model.eval()
    with torch.no_grad():
        y_pred = torch.zeros((X.shape[0], model.output_dim))
        for j in range(0, X.shape[0], batch_size):
            X_batch = X[j : j + batch_size]
            y_pred[j : j + batch_size] = model(X_batch)
    return y_pred
