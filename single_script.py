# %%
"""Probabilistic regression in 1D"""
import time
from pathlib import Path
import numpy as np
import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch import nn
import torch


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

    @abstractmethod
    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        pass

    @property
    @abstractmethod
    def name(self):
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

    def cdf(
        self,
        predicted_params: torch.Tensor,
        extra_params: dict,
        points_to_evaluate: torch.Tensor,
    ):
        # predicted params (B, 1), extra_params['std'] (1), points_to_evaluate (N)
        dist = torch.distributions.Normal(predicted_params, extra_params["std"])
        return dist.cdf(points_to_evaluate)  # output shape (B, N)

    @property
    def name(self):
        return "Mean Regressor"


class MedianRegressor(Regressor):
    def __init__(self):
        self.mlp = SimpleMLP()
        self.b = None

    def compute_extra_params(self, X, y):
        with torch.no_grad():
            b = torch.median(torch.abs(self.mlp(X) - torch.from_numpy(y)))
        return {"b": b}

    def loss_fn(self, predicted_params, y_target):
        return torch.mean(torch.abs(predicted_params - y_target))

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        dist = torch.distributions.Laplace(predicted_params, extra_params["b"])
        return dist.cdf(points_to_evaluate)

    @property
    def name(self):
        return "Median Regressor"


class MeanStdRegressor(Regressor):
    def __init__(self):
        self.mlp = SimpleMLP(output_size=2)

    def loss_fn(self, predicted_params, y_target):
        mu, sigma2 = predicted_params[:, 0:1], torch.nn.functional.softplus(
            predicted_params[:, 1:2]
        )
        return (
            0.5 * (torch.log(2 * torch.pi * sigma2) + (mu - y_target) ** 2 / sigma2)
        ).mean()

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        mu, sigma2 = predicted_params[:, 0:1], torch.nn.functional.softplus(
            predicted_params[:, 1:2]
        )
        dist = torch.distributions.Normal(mu, torch.sqrt(sigma2))
        return dist.cdf(points_to_evaluate)

    @property
    def name(self):
        return "MeanStd Regressor"


class MedianScaleRegressor(Regressor):
    def __init__(self):
        self.mlp = SimpleMLP(output_size=2)

    def loss_fn(self, predicted_params, y_target):
        median, scale = predicted_params[:, 0:1], torch.nn.functional.softplus(
            predicted_params[:, 1:2]
        )
        return (torch.log(2 * scale) + torch.abs(median - y_target) / scale).mean()

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        median, scale = predicted_params[:, 0:1], torch.nn.functional.softplus(
            predicted_params[:, 1:2]
        )
        dist = torch.distributions.Laplace(median, scale)
        return dist.cdf(points_to_evaluate)

    @property
    def name(self):
        return "MedianScale Regressor"


class QuantileRegressor(Regressor):
    def __init__(self, n_quantiles=10):
        self.mlp = SimpleMLP(output_size=n_quantiles + 1)
        self.n_quantiles = n_quantiles
        self.taus = torch.linspace(0, 1, self.n_quantiles + 1)  # (nq+1)

    def loss_fn(self, predicted_params, y_target):
        loss = (
            (
                (self.taus[None] - 1 * (y_target <= predicted_params))
                * (y_target - predicted_params)
            )
            .sum(dim=1)
            .mean()
        )
        return loss

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        # (B, nq+1), None, (N)
        B, nq, N = (
            predicted_params.shape[0],
            predicted_params.shape[1] - 1,
            points_to_evaluate.shape[0],
        )
        distances = torch.diff(predicted_params, dim=1).reshape(B, nq, 1)  # (B, nq, 1)
        relus = torch.nn.functional.relu(
            points_to_evaluate.reshape(1, 1, N) - predicted_params.reshape(B, nq + 1, 1)
        )  # (B, nq+1, N)  memory-intense
        ramps_up = relus[:, :-1]  # (B, nq, N)
        ramps_down = relus[:, 1:]  # (B, nq, N)
        return (((ramps_up - ramps_down) / distances) / self.n_quantiles).sum(
            dim=1
        )  # (B, N)  memory-intense

    @property
    def name(self):
        return "Quantile Regressor"


class ImplicitNetwork(nn.Module):
    def __init__(self, d=32, n=64):
        super(ImplicitNetwork, self).__init__()
        self.d, self.n = d, n
        self.encoder = SimpleMLP(output_size=d)  # feature extractor
        self.mlp2 = SimpleMLP(input_size=d, output_size=1, hidden_sizes=[4 * d] * 3)
        self.embedding_projector = nn.Linear(n, d)

    def embed(self, taus):
        # taus (T)
        cosemb = torch.cos(
            torch.pi * torch.arange(self.n)[None] * taus.reshape(-1, 1)
        )  # (T, n)
        emb_taus = torch.nn.functional.relu(self.embedding_projector(cosemb))  # (T, d)
        return emb_taus

    def forward(self, x):
        return self.encoder(x)  # (B, d)

    def forward2(self, x, taus):
        # (B, F), (T)
        B, F, T = x.shape[0], x.shape[1], taus.shape[0]
        xf = self.encoder(x)  # (B, d)
        emb_taus = self.embed(taus)  # (T, d)
        feats = xf.reshape(B, 1, self.d) * emb_taus.reshape(1, T, self.d)  # (B, T, d)
        preds = self.mlp2(feats)  # (B, T, 1)
        return preds


class ImplicitQuantileRegressor(Regressor):
    def __init__(self, d=32, n=64, N=8):
        self.d = d  # feature dimension
        self.n = n  # cosine embedding dimension
        self.N = N  # samples to use at the loss
        self.mlp = ImplicitNetwork(d=d, n=n)  # call it mlp for compatibility

    def loss_fn(self, predicted_params, y_target):
        # predicted params (B, d)
        self.mlp.train()
        random_taus = torch.rand(self.N)
        B, N = predicted_params.shape[0], random_taus.shape[0]
        xf = predicted_params  # (B, d)
        emb_taus = self.mlp.embed(random_taus)  # (N, d)
        feats = xf.reshape(B, 1, self.d) * emb_taus.reshape(1, N, self.d)  # (B, N, d)
        preds = self.mlp.mlp2(feats.reshape(B * N, self.d)).reshape(B, N)  # (B, N)
        loss = (  # quantile loss
            (
                (random_taus[None] - 1 * (y_target <= preds)) * (y_target - preds)
            )  # (B, N)
            .sum(dim=1)
            .mean()
        )
        return loss

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        self.mlp.eval()
        with torch.no_grad():
            P = len(points_to_evaluate)
            many_taus = torch.linspace(0, 1, min(100, P))
            B, N = predicted_params.shape[0], many_taus.shape[0]
            xf = predicted_params  # (B, d)
            emb_taus = self.mlp.embed(many_taus)  # (N, d)
            feats = xf.reshape(B, 1, self.d) * emb_taus.reshape(
                1, N, self.d
            )  # (B, N, d)
            preds = self.mlp.mlp2(feats.reshape(B * N, self.d)).reshape(B, N)  # (B, N)
            preds = torch.sort(
                preds, dim=1
            ).values  # (B, N)  order to make monotonically increasing, it's an smoothing
            is_higher = preds.reshape(B, N, 1) >= points_to_evaluate.reshape(
                1, 1, P
            )  # (B, N, P) see if the predicted are higher than the points to evaluate
            is_lower = ~is_higher  # (B, N, P)
            preds_higher_masked = torch.where(
                is_higher, preds.reshape(B, N, 1), preds.max() + 1
            )  # (B, N, P) keep those points that are higher at their original value
            preds_lower_masked = torch.where(
                is_lower, preds.reshape(B, N, 1), preds.min() - 1
            )  # (B, N, P)
            nearest_higher = preds_higher_masked.min(
                dim=1
            ).values  # (B, P)  the nearest higher value for each point
            nearest_lower = preds_lower_masked.max(dim=1).values  # (B, P)
            higher_indices = (
                (preds.reshape(B, N, 1) == nearest_higher.reshape(B, 1, P))
                .max(dim=1)
                .indices
            )  # (B, P)  the indices of the nearest higher value
            lower_indices = (
                (preds.reshape(B, N, 1) == nearest_lower.reshape(B, 1, P))
                .max(dim=1)
                .indices
            )  # (B, P)
            taus_higher = many_taus[higher_indices.reshape(B * P)].reshape(B, P)
            taus_lower = many_taus[lower_indices.reshape(B * P)].reshape(B, P)
            # linear interpolation
            weight = (points_to_evaluate.reshape(1, P) - nearest_lower) / (
                nearest_higher - nearest_lower
            )  # (B, P)
            interpolated_probs = taus_lower + weight * (
                taus_higher - taus_lower
            )  # (B, P)
        return interpolated_probs

    @property
    def name(self):
        return "Implicit Quantile Regressor"


class BinClassifierRegressor(Regressor):
    def __init__(self, n_bins=20, lower=-6, upper=6):
        self.n_bins = n_bins
        self.lower = lower
        self.upper = upper
        self.mlp = SimpleMLP(output_size=n_bins)
        self.bin_edges = torch.cat(
            [
                -torch.tensor([torch.inf]),
                torch.linspace(lower, upper, n_bins - 1),
                torch.tensor([torch.inf]),
            ]
        )
        self.distance = self.bin_edges[2] - self.bin_edges[1]

    def loss_fn(self, predicted_params, y_target):
        # y_target (B, 1)
        # cum_one_hot = 1 * (y_target >= self.bin_edges[None, :-1])  # (B, n_bins)
        assert (
            self.bin_edges[0] <= y_target.min() and y_target.max() <= self.bin_edges[-1]
        ), f"y_target out of bounds: {y_target.min()} {y_target.max()}, redefine lower and upper"
        bin_indices = (
            torch.bucketize(y_target, self.bin_edges, right=True) - 1
        )  # (B, 1)
        one_hot_encoded = torch.zeros(len(y_target), self.n_bins).scatter(
            1, bin_indices, 1
        )  # (B, n_bins)
        return torch.nn.functional.cross_entropy(
            predicted_params, one_hot_encoded, reduction="mean"
        )

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        B, n_bins = predicted_params.shape
        probs = predicted_params.softmax(dim=1)
        cum_probs = probs.cumsum(dim=1)

        points_to_evaluate_as_bin_inds = torch.clamp(
            ((points_to_evaluate - self.lower) / self.distance).floor() + 1,
            min=0,
            max=n_bins - 1,
        ).long()  # Ensure integer indices
        remainders = (
            (torch.clamp(points_to_evaluate, max=self.upper) - self.lower)
            % self.distance
        ) * (points_to_evaluate >= self.lower) + (
            points_to_evaluate < self.lower
        ) * 1  # (P)

        cumprob_until_prev_bin = torch.cat([torch.zeros((B, 1)), cum_probs], dim=1)[
            :, points_to_evaluate_as_bin_inds
        ]
        excess_of_current_bin = (
            probs[:, points_to_evaluate_as_bin_inds] * remainders[None]
        )  # Adjusted for broadcasting
        cumprobs = cumprob_until_prev_bin + excess_of_current_bin

        return cumprobs

    @property
    def name(self):
        return "Bin Classifier Regressor"


def fit_torch(
    model,
    X,
    y,
    loss_fn,
    batch_size=16384,
    lr=1e-1,
    n_epochs=24,
    optim="adamw",
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


def evaluate(
    reg, Xtest, ytest, lower=-15, upper=15, n_bins=int(1e4), plot_to="", batch_size=128
):
    points_to_evaluate = torch.linspace(
        lower, upper, n_bins
    )  # evaluate cdf at these points
    # prediction
    cum_probs = []
    for i in tqdm.tqdm(range(0, len(Xtest), batch_size)):
        Xbatch = Xtest[i : i + batch_size]
        pred_params, extra_params = reg.predict(Xbatch)
        cum_probs_batch = reg.cdf(
            pred_params, extra_params, points_to_evaluate
        )  # B, n_bins  # cumulative probability at each point
        cum_probs.append(cum_probs_batch)
    cum_probs = torch.cat(cum_probs, dim=0)
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
    mae = torch.abs(median_deterministic_predictions - ytest).mean()
    string = "\n".join(
        [
            ("-" * 40),
            (f"CRP error: {crpes.mean().item():.3e}"),
            (f"Calibration error: {calibration_error.item():.3e}"),
            (f"Centering error: {centering_error.item():.3e}"),
            (f"MAE of the median predictor: {mae.item():.3e}"),
            ("-" * 40),
        ]
    )

    if plot_to:
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
            (cum_probs_at_y[None] <= xticks[:, None]).sum(dim=1) / len(cum_probs_at_y),
            label="Empirical",
        )
        plt.plot(np.linspace(0, 1, 101), np.linspace(0, 1, 101), "k--", label="Ideal")
        plt.legend()
        plt.xlabel("Assigned cumulative probability")
        plt.ylabel("Empirical cumulative probability")

        plt.tight_layout()
        plt.savefig(plot_to)
        plt.close()

    print(string)
    return string
    # return {
    #     "crpes": crpes,
    #     "calibration_error": calibration_error,
    #     "centering_error": centering_error,
    #     "mae": mae,
    # }


# %%
def main(
    SEED=0,
    N=10000,
    N_test=100,
    mean1=0,
    mean2=1,
    std1=1.1,
    std2=0.1,
    n_epochs=144,
    n_factors=10,
    postfix="",
):
    args = locals()
    mixture_factors = np.linspace(0, 1, n_factors)
    # %%
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    X, y = generate_data(N, mixture_factors, mean1, mean2, std1, std2)
    Xtest, ytest = generate_data(N_test, mixture_factors, mean1, mean2, std1, std2)
    Path("runs").mkdir(exist_ok=True, parents=True)
    # %%
    for reg in [
        MeanRegressor(),
        MedianRegressor(),
        MeanStdRegressor(),
        MedianScaleRegressor(),
        BinClassifierRegressor(),
        QuantileRegressor(),
        ImplicitQuantileRegressor(),
    ]:
        print(f"Training {reg.name}")
        st = time.time()
        reg = reg.fit(X, y, n_epochs=n_epochs)
        train_time = time.time() - st
        print(f"Training took {train_time:.2f} seconds")
        print(f"Evaluating {reg.name}")
        st = time.time()
        string = evaluate(reg, Xtest, ytest, plot_to=f"runs/{reg.name+postfix}.png")
        eval_time = time.time() - st
        with open(f"runs/{reg.name+postfix}.txt", "w") as f:
            f.write(f"args = {args}" + "\n")
            f.write(f"-" * 50 + "\n")
            f.write(f"{reg.name}" + "\n")
            f.write(f"Training took {train_time} seconds" + "\n")
            f.write(f"Evaluating took {eval_time} seconds" + "\n")
            f.write(string + "\n")
        print(f"Evaluating took {eval_time} seconds")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
