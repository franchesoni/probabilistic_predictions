# %%
"""Probabilistic regression in 1D"""
import numpy as np
import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch import nn
import torch
from scipy.integrate import quad


def generate_data(N, mixture_factors, mean1, mean2, std1, std2):
    """Simulate sampling from a gaussian mixture by sampling N*factor points from the first gaussian and N*(1-factor) points from the second gaussian."""
    points = []
    for mixture_factor in mixture_factors:
        N1 = int(N * mixture_factor)
        N2 = N - N1
        points1 = norm.rvs(mean1, std1, size=N1)
        points2 = norm.rvs(mean2, std2, size=N2)
        points.append(np.concatenate([points1, points2]))
    y = np.array(points)  # targets
    X = np.array([np.ones(N) * mf for mf in mixture_factors])
    return X.reshape(-1, 1), y.reshape(-1, 1)


# %%
# The objective is to learn a function that given the input (in [0,1]) outputs a probability distribution.
# This is, make a function learn to output functions
# The function will be composed by a model and a loss
# Here we define the model as the simplest mlp
class SimpleMLP(nn.Module):
    def __init__(
        self, input_size=1, hidden_sizes=[128, 64], output_size=1, dropout_rate=0.1
    ):
        super(SimpleMLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))

            if i < len(sizes) - 2:  # No activation or normalization on the output layer
                layers.append(
                    nn.BatchNorm1d(sizes[i + 1])
                )  # Batch normalization for stability
                layers.append(nn.GELU())  # GELU activation
                layers.append(nn.Dropout(dropout_rate))  # Dropout for regularization

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.model(x)


# %%
from abc import ABC, abstractmethod


class Regressor(ABC):
    def fit(self, X, y, n_epochs=1):
        self.mlp = fit_torch(self.mlp, X, y, self.loss_fn, n_epochs=n_epochs)
        self.extra = self.compute_extra_params(X, y)
        return self

    def compute_extra_params(self, X, y):
        return None

    def predict(self, X):
        with torch.no_grad():
            self.mlp.eval()
            predicted_params = self.mlp(X)
        extra_params = self.extra
        return predicted_params, extra_params

    @abstractmethod
    def loss_fn(self, predicted_params, y_target):
        pass

    @staticmethod
    @abstractmethod
    def cdf(predicted_params, extra_params, points_to_evaluate):
        pass


# Now define our first candidate. This is an instance that has a .fit method and that when given a value it outputs a probability distribution.
class MeanRegressor(Regressor):
    def __init__(self):
        self.mlp = SimpleMLP()
        self.std = None

    def compute_extra_params(self, X, y):
        with torch.no_grad():
            std = torch.sqrt(torch.mean((self.mlp(X) - torch.from_numpy(y)) ** 2))
        return {"std": std}

    def loss_fn(self, predicted_params, y_target):
        # the predicted params is simply the mean
        return torch.mean(
            (predicted_params - y_target) ** 2
        )  # Mean squared error which is equivalent to the variance

    @staticmethod
    def cdf(
        predicted_params: torch.Tensor,
        extra_params: dict,
        points_to_evaluate: torch.Tensor,
    ):
        # predicted params (B, 1), extra_params['std'] (1), points_to_evaluate (N)
        dist = torch.distributions.Normal(predicted_params, extra_params["std"])
        return dist.cdf(points_to_evaluate)  # output shape (B, N)


# class MedianRegressor:
#     def __init__(self):
#         self.mlp = SimpleMLP()
#         self.b = None

#     def fit(self, X, y, n_epochs=1):
#         self.mlp = fit_torch(self.mlp, X, y, self.loss_fn, n_epochs=n_epochs)
#         with torch.no_grad():
#             self.b = torch.median(torch.abs(self.mlp(X) - torch.from_numpy(y)))
#         return self

#     def loss_fn(self, predicted_params, y_target):
#         return torch.mean(torch.abs(predicted_params- y_target))

#     def predict_cdf(self, x):
#         self.mlp.eval()
#         with torch.no_grad():
#             predicted_params = self.mlp(x[None])
#             dist = torch.distributions.Laplace(predicted_params, self.b)
#         def cdf_func(point_to_evaluate):
#             return dist.cdf(torch.Tensor([point_to_evaluate])).item()
#         return cdf_func, predicted_params - 5 * self.b, predicted_params + 5 * self.b

# class MeanStdRegressor:
#     def __init__(self):
#         self.mlp = SimpleMLP(output_size=2)

#     def fit(self, X, y, n_epochs=1):
#         self.mlp = fit_torch(self.mlp, X, y, self.loss_fn, n_epochs=n_epochs)
#         return self

#     def loss_fn(self, predicted_params, y_target):
#         mu, sigma2 = predicted_params[:, 0], torch.nn.functional.softplus(predicted_params[:, 1])
#         return 0.5 * (torch.log(2 * torch.pi * sigma2) + (mu - y_target) ** 2 / sigma2)

#     def predict_cdf(self, x):
#         self.mlp.eval()
#         with torch.no_grad():
#             predicted_params = self.mlp(x[None])
#             mu, sigma2 = predicted_params[:, 0], torch.nn.functional.softplus(predicted_params[:, 1])
#             dist = torch.distributions.Normal(predicted_params[:, 0], torch.sqrt(predicted_params[:, 1]))
#         def cdf_func(point_to_evaluate):
#             return dist.cdf(torch.Tensor([point_to_evaluate])).item()
#         return cdf_func, mu - torch.sqrt(sigma2) * 3.5, mu + torch.sqrt(sigma2) * 3.5

# class MedianScaleRegressor:
#     def __init__(self):
#         self.mlp = SimpleMLP(output_size=2)

#     def fit(self, X, y, n_epochs=1):
#         self.mlp = fit_torch(self.mlp, X, y, self.loss_fn, n_epochs=n_epochs)
#         return self

#     def loss_fn(self, predicted_params, y_target):
#         median, scale = predicted_params[:, 0], torch.nn.functional.softplus(predicted_params[:, 1])
#         return torch.log(2 * scale) + torch.abs(median - y_target) / scale

#     def predict_cdf(self, x):
#         self.mlp.eval()
#         with torch.no_grad():
#             predicted_params = self.mlp(x[None])
#             median, scale = predicted_params[:, 0], torch.nn.functional.softplus(predicted_params[:, 1])
#             dist = torch.distributions.Laplace(median, scale)
#         def cdf_func(point_to_evaluate):
#             return dist.cdf(torch.Tensor([point_to_evaluate])).item()
#         return cdf_func, median - 5 * scale, median + 5 * scale

# class QuantileRegressor:
#     def __init__(self, n_quantiles=4):
#         self.mlp = SimpleMLP(output_size=n_quantiles+1)
#         self.n_quantiles = n_quantiles
#         self.taus = torch.linspace(0, 1, self.n_quantiles+1)  # (nq+1)

#     def fit(self, X, y, n_epochs=1):
#         self.mlp = fit_torch(self.mlp, X, y, self.loss_fn, n_epochs=n_epochs)
#         return self

#     def loss_fn(self, predicted_params, y_target):
#         assert predicted_params.shape[0] == 1 and predicted_params.shape[1] == len(self.taus)
#         loss = ((self.taus[None] - (predicted_params <= y_target)) *
#                 (y_target - predicted_params)).mean()
#         return loss

#     def predict_cdf(self, x):
#         self.mlp.eval()
#         with torch.no_grad():
#             predicted_params = self.mlp(x[None])
#             distances = torch.diff(predicted_params)
#         def cdf_func(point_to_evaluate):
#             relus = torch.nn.functional.relu(point_to_evaluate - predicted_params)  # nq+1
#             ramps_up = relus[:-1]  # nq
#             ramps_down = - relus[1:]  # nq
#             return (((ramps_up + ramps_down) / distances) / self.n_quantiles).sum()
#         return cdf_func


def fit_torch(
    model,
    X,
    y,
    loss_fn,
    batch_size=16384,
    lr=1e-1,
    n_epochs=24,
    optim="sgd",
    verbose=True,
    extra_metrics={},
    extra_metrics_every=1,
    extra_datalimit=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Fit a torch model using gradient descent."""
    X, y = torch.from_numpy(X).float().to(device), torch.from_numpy(y).float().to(
        device
    )
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=lr)
        if optim == "adamw"
        else (
            torch.optim.SGD(model.parameters(), lr=lr)
            if optim == "sgd"
            else NotImplementedError(f"optimizer {optim} not implemented")
        )
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=X.shape[0] // batch_size + 1,
        epochs=n_epochs,
    )
    model.train()
    with tqdm.tqdm(range(n_epochs)) as pbar:  # epochs
        for i in pbar:
            pbar.set_description(f"Epoch {i+1}/{n_epochs}")
            cumloss = 0
            for j in range(0, X.shape[0], batch_size):  # batches
                X_batch = X[j : j + batch_size]
                y_batch = y[j : j + batch_size]
                optimizer.zero_grad()
                predicted_params = model(X_batch)
                loss = loss_fn(predicted_params, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                cumloss += loss.item()
            cumloss /= X.shape[0] / batch_size
            pbar.set_postfix({"loss": f"{cumloss:.4f}"})
            if i % extra_metrics_every == 0:
                with torch.no_grad():
                    extras = {}
                    for name, metric in extra_metrics.items():
                        extras[name] = metric(
                            predicted_params[:extra_datalimit],
                            y_batch[:extra_datalimit],
                        )
    return model


def crps_single_prediction(
    predicted_cdf, ground_truth, lower_bound, upper_bound, eps=1e-03
):
    """
    Compute the CRPS for a single prediction.

    :param predicted_cdf: Function that computes the predicted CDF at a given point.
    :param ground_truth: The observed ground truth value.
    :param lower_bound: Lower bound for integration.
    :param upper_bound: Upper bound for integration.
    :return: CRPS for the given prediction.
    """

    def integrand(x):
        return (predicted_cdf(x) - 1 * (x >= ground_truth)) ** 2

    crps, _ = quad(
        integrand, lower_bound, upper_bound, epsabs=eps, epsrel=eps, limit=15
    )
    return crps


def batch_crps(
    points_to_evaluate, cum_probs, y, lower_bound, upper_bound, n_bins=int(1e4)
):
    """Computes the CRPS for a regressor"""
    heavyside = 1 * (points_to_evaluate[None] >= torch.from_numpy(y))  # B, n_bins
    crpss = (
        ((cum_probs - heavyside) ** 2).sum(dim=1) / n_bins * (upper_bound - lower_bound)
    )  # B
    # now compute the assigned cum prob to each ground truth
    return crpss


def evaluate(reg, Xtest, ytest, lower=-15, upper=15, n_bins=int(1e4)):
    # prediction
    pred_params, extra_params = reg.predict(Xtest)
    points_to_evaluate = torch.linspace(
        lower, upper, n_bins
    )  # evaluate cdf at these points
    cum_probs = reg.cdf(
        pred_params, extra_params, points_to_evaluate
    )  # B, n_bins  # cumulative probability at each point
    # aux variables
    median_indices = torch.argmin(
        torch.abs(cum_probs - 0.5), dim=1
    )  # the indices of the median
    y_indices = torch.argmin(
        torch.abs(points_to_evaluate[None] - torch.from_numpy(ytest)), dim=1
    )  # the indices of the evaluated points that correspond to the ground truth
    cum_probs_at_y = torch.gather(
        cum_probs, 1, y_indices[:, None]
    ).squeeze()  # the cumulative probability assigned to the ground truth
    median_deterministic_predictions = points_to_evaluate[
        median_indices
    ]  # the median prediction

    # evaluation
    crpes = batch_crps(
        points_to_evaluate, cum_probs, ytest, lower, upper, n_bins
    )  # the crps for each prediction
    calibration_error = torch.abs(
        torch.argsort(cum_probs_at_y) / len(cum_probs_at_y) - cum_probs_at_y
    ).mean()
    centering_error = (cum_probs_at_y - 0.5).abs().mean()
    print(f"Calibration error: {calibration_error.item():.3e}")
    print(f"CRP error: {crpes.mean().item():.3e}")
    print(f"Centering error: {centering_error.item():.3e}")

    plt.figure()

    plt.subplot(211)  # PIT hist
    plt.hist(cum_probs_at_y, bins=20, density=True, label="Empirical")
    plt.plot(np.linspace(0, 1, 101), np.ones(101), "k--", label="Ideal")
    plt.legend()
    plt.xlabel("Assigned cumulative probability")
    plt.ylabel("Empirical Density")

    plt.subplot(212)  # reliability diagram
    xticks = torch.linspace(0, 1, 101)
    plt.plot(
        xticks,
        (xticks[:, None] >= cum_probs_at_y[None]).sum(dim=1) / len(cum_probs_at_y),
        label="Empirical",
    )
    plt.plot(np.linspace(0, 1, 101), np.linspace(0, 1, 101), "k--", label="Ideal")
    plt.legend()
    plt.xlabel("Assigned cumulative probability")
    plt.ylabel("Empirical cumulative probability")

    plt.tight_layout()
    plt.show()

    return {
        "crpes": crpes,
        "calibration_error": calibration_error,
        "centering_error": centering_error,
    }


# %%
SEED = 0
N = 10000
mean1, mean2, std1, std2 = 0, 1, 1.1, 0.1
mixture_factors = np.linspace(0, 1, 10)
# %%
np.random.seed(SEED)
torch.manual_seed(SEED)
X, y = generate_data(N, mixture_factors, mean1, mean2, std1, std2)
Xtest, ytest = generate_data(1000, mixture_factors, mean1, mean2, std1, std2)
# %%
reg = MeanRegressor()
n_epochs = 4
reg = reg.fit(X, y, n_epochs=n_epochs)
metrics = evaluate(reg, Xtest, ytest)

# %%
