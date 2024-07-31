from typing import Sequence
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

log2pi = torch.log(torch.tensor(2 * torch.pi))


class MLP(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], activation_fn=nn.GELU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ProbabilisticMethod(ABC):
    @abstractmethod
    def get_F_at_y(self, batch_y, pred_params):
        pass

    @abstractmethod
    def get_logscore_at_y(self, batch_y, pred_params):
        pass

    @abstractmethod
    def loss(self, batch_y, pred_params):
        pass

    @abstractmethod
    def is_PL(self) -> bool:
        pass

    def forward(self, batch_x):
        pass


class LaplaceLogScore(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(LaplaceLogScore, self).__init__()
        self.model = MLP(layer_sizes + [2], **kwargs)

    def get_F_at_y(self, batch_y, pred_params):
        mus, bs = pred_params[:, 0:1], pred_params[:, 1:]
        return 0.5 + 0.5 * (2 * (mus < batch_y) - 1) * (
            1 - torch.exp(-torch.abs(mus - batch_y) / bs)
        )

    def get_logscore_at_y(self, batch_y, pred_params):
        mus, bs = pred_params[:, 0:1], pred_params[:, 1:]
        return torch.log(2 * bs) + torch.abs(mus - batch_y) / bs

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(batch_y, pred_params).sum()

    def is_PL(self) -> bool:
        return False

    def forward(self, batch_x):
        params = self.model(batch_x)  # logits
        mu, blogits = params[:, 0], params[:, 1]
        blogits = nn.functional.softplus(blogits)  # w_b must be positive
        return torch.stack([mu, blogits], dim=1)


class LaplaceGlobalWidth(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, train_width=True, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(LaplaceGlobalWidth, self).__init__()
        self.model = MLP(layer_sizes + [1], **kwargs)
        self.global_width = torch.tensor([1.0])
        self.train_width = train_width
        if train_width:
            self.global_width = nn.Parameter(self.global_width)

    def get_F_at_y(self, batch_y, pred_params):
        mus = pred_params
        bs = self.global_width
        return 0.5 + 0.5 * (2 * (mus < batch_y) - 1) * (
            1 - torch.exp(-torch.abs(mus - batch_y) / bs)
        )

    def get_logscore_at_y(self, batch_y, pred_params):
        mus, bs = pred_params, self.global_width
        assert bs > 0, "Width must be positive"
        return torch.log(2 * bs) + torch.abs(mus - batch_y) / bs

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(batch_y, pred_params).sum()

    def is_PL(self) -> bool:
        return False

    def forward(self, batch_x):
        return self.model(batch_x)


class MixtureDensityNetwork(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, n_components, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(MixtureDensityNetwork, self).__init__()
        self.model = MLP(layer_sizes + [3 * n_components], **kwargs)
        self.n_components = n_components

    def get_F_at_y(self, batch_y, pred_params):
        pis, mus, sigmas = (
            pred_params[:, : self.n_components],
            pred_params[:, self.n_components : 2 * self.n_components],
            pred_params[:, 2 * self.n_components :],
        )
        assert pis.shape[0] == batch_y.shape[0]
        assert len(batch_y.shape) == 1
        assert pis.shape[1] == self.n_components
        batch_y = batch_y.reshape(-1, 1)
        return (
            pis * 0.5 * (1 + torch.erf((batch_y - mus) / (sigmas * (2**0.5))))
        ).sum(dim=1)

    def get_logscore_at_y(self, batch_y, pred_params):
        pis, mus, sigmas = (
            pred_params[:, : self.n_components],
            pred_params[:, self.n_components : 2 * self.n_components],
            pred_params[:, 2 * self.n_components :],
        )
        assert pis.shape[0] == batch_y.shape[0]
        assert pis.shape[1] == self.n_components
        assert len(batch_y.shape) == 2
        batch_y = batch_y.unsqueeze(1)  # (B, 1, Y)
        pis, mus, sigmas = (
            pis.unsqueeze(2),
            mus.unsqueeze(2),
            sigmas.unsqueeze(2),
        )  # (B, K, 1)
        log_prob_per_component = (
            -0.5 * (((batch_y - mus) / sigmas) ** 2) - torch.log(sigmas) - 0.5 * log2pi
        )  # (B, K, Y)
        weighted_log_prob = torch.log(pis) + log_prob_per_component
        neg_log_likelihood = -torch.logsumexp(weighted_log_prob, dim=1)  # (B, Y)
        return neg_log_likelihood

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(batch_y, pred_params).sum()

    def is_PL(self) -> bool:
        return False

    def forward(self, batch_x):
        params = self.model(batch_x)  # logits
        pis, mus, sigmas = (
            params[:, : self.n_components],
            params[:, self.n_components : 2 * self.n_components],
            params[:, 2 * self.n_components :],
        )
        pis = nn.functional.softmax(pis, dim=1)  # pis must be positive and sum to 1
        sigmas = nn.functional.softplus(sigmas)  # w_b must be positive
        return torch.concatenate([pis, mus, sigmas], dim=1)


class CategoricalCrossEntropy(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, bin_borders, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(CategoricalCrossEntropy, self).__init__()
        self.B = len(bin_borders) - 1
        self.bin_borders = torch.tensor(bin_borders)
        self.bin_widths = self.bin_borders[1:] - self.bin_borders[:-1]
        assert self.bin_widths.min() > 0, "Bin borders must be strictly increasing"
        self.model = MLP(layer_sizes + [self.B], **kwargs)

    def get_F_at_y(self, batch_y, pred_params):
        N, B = pred_params.shape
        cdf = torch.cat(
            (
                torch.zeros((N, 1), device=pred_params.device),
                torch.cumsum(pred_params, dim=1),
            ),
            dim=1,
        )
        # bin index for each y
        k = torch.clamp(
            torch.searchsorted(self.bin_borders, batch_y.reshape(N, 1)) - 1, 0, B - 1
        ).squeeze()
        # Compute CDF at y using linear interpolation
        cdf_at_y = cdf[torch.arange(N), k] + (
            (
                (batch_y - self.bin_borders[k])
                / (self.bin_widths[k])
                * (cdf[torch.arange(N), k + 1] - cdf[torch.arange(N), k])
            )
            * (1 * (0 < self.bin_widths[k]))
        )
        return cdf_at_y

    def get_logscore_at_y(self, batch_y, pred_params):
        return get_logscore_at_y_PL(
            batch_y,
            all_quantiles=None,
            bin_masses=pred_params,
            bin_borders=self.bin_borders.reshape(1, -1),
        )

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(batch_y, pred_params).sum()

    def is_PL(self) -> bool:
        return True

    def forward(self, batch_x):
        logits = self.model(batch_x)
        masses = nn.functional.softmax(logits, dim=1)
        return masses


class PinballLoss(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, quantile_levels, bounds, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(PinballLoss, self).__init__()
        self.quantile_levels = torch.sort(torch.tensor(quantile_levels))[0]
        assert self.quantile_levels.min() > 0, "Quantiles must be in [0, 1]"
        assert self.quantile_levels.max() < 1, "Quantiles must be in [0, 1]"
        self.lower, self.upper = bounds
        self.quantile_levels = torch.concatenate(
            (torch.tensor([0]), self.quantile_levels, torch.tensor([1]))
        )
        self.B = len(self.quantile_levels) - 1
        self.model = MLP(layer_sizes + [len(quantile_levels)], **kwargs)

    def get_F_at_y(self, batch_y, pred_params):
        N, Q = pred_params.shape  # Q = number of quantiles
        cdf = self.quantile_levels.expand(N, self.B + 1)
        bin_borders = torch.concatenate(
            (
                torch.tensor(self.lower).reshape(1, 1).expand(N, 1),
                pred_params,
                torch.tensor(self.upper).reshape(1, 1).expand(N, 1),
            ),
            dim=1,
        )
        # bin index for each y
        k = torch.clamp(
            torch.searchsorted(bin_borders, batch_y.reshape(N, 1)) - 1, 0, self.B - 1
        ).squeeze()
        # Compute CDF at y using linear interpolation
        cdf_at_y = cdf[torch.arange(N), k] + (
            (
                (batch_y - bin_borders[torch.arange(N), k])
                / (
                    bin_borders[torch.arange(N), k + 1]
                    - bin_borders[torch.arange(N), k]
                )
                * (cdf[torch.arange(N), k + 1] - cdf[torch.arange(N), k])
            )
            * (
                1
                * (
                    bin_borders[torch.arange(N), k + 1]
                    > bin_borders[torch.arange(N), k]
                )
            )
        )
        return cdf_at_y

    def get_logscore_at_y(self, batch_y, pred_params):
        N = batch_y.shape[0]
        bin_borders = torch.concatenate(
            (
                torch.tensor(self.lower).reshape(1, 1).expand(N, 1),
                pred_params,
                torch.tensor(self.upper).reshape(1, 1).expand(N, 1),
            ),
            dim=1,
        )
        return get_logscore_at_y_PL(
            batch_y,
            all_quantiles=self.quantile_levels.reshape(1, -1),
            bin_masses=None,
            bin_borders=bin_borders,
        )

    def loss(self, batch_y, pred_params):
        batch_y = batch_y.reshape(-1, 1)
        pinball = (self.quantile_levels[1:-1] - 1 * (batch_y < pred_params)) * (
            batch_y - pred_params
        )
        return pinball.sum(dim=1).mean()

    def is_PL(self) -> bool:
        return True

    def forward(self, batch_x):
        return self.model(batch_x)


def get_logscore_at_y_PL(batch_y, all_quantiles, bin_masses, bin_borders):
    assert (bin_masses is None and all_quantiles is not None) or (
        bin_masses is not None and all_quantiles is None
    )
    N, Y = batch_y.shape
    assert (all_quantiles is None and bin_masses.shape[0] in [1, N]) or (
        bin_masses is None and all_quantiles.shape[0] in [1, N]
    )
    assert bin_borders.shape[0] in [1, N]
    assert bin_borders.shape[1] == (
        all_quantiles.shape[1] if all_quantiles is not None else bin_masses.shape[1] + 1
    )
    B = all_quantiles.shape[1] - 1 if all_quantiles is not None else bin_masses.shape[1]
    # the log score of a histogram is the neg log of the pdf
    # the pdf is the mass of the bin divided by the bin width
    if bin_masses is None:
        bin_masses = all_quantiles[:, 1:] - all_quantiles[:, :-1]  # (1N, B)
    bin_widths = bin_borders[:, 1:] - bin_borders[:, :-1]  # (1N, B)
    bin_densities = bin_masses / bin_widths  # (1N, B)
    y_bin = torch.clamp(
        torch.searchsorted(bin_borders.squeeze(), batch_y, right=True) - 1, 0, B - 1
    )  # here we squeeze bin borders because when it has leading shape N it's fine, but when it has leading shape 1 it's not (we need to make it 1d in that case)
    # reshape and compute
    bin_densities = bin_densities.reshape(N, B, 1)  # (N, 1, B)
    y_bin = y_bin.reshape(N, 1, y_bin.shape[1])  # (N, 1, Y)
    log_score = -(
        torch.log(bin_densities + 1e-45) * (y_bin == torch.arange(B).reshape(1, B, 1))
    ).sum(dim=1)
    return log_score


def test_laplace():
    torch.autograd.set_detect_anomaly(True)
    for fixed_pred_params, method in [
        (torch.tensor([[0.2, 0.5]]), LaplaceLogScore([12, 128, 128])),
        (torch.tensor([[0.2]]), LaplaceGlobalWidth([12, 128, 128])),
        (torch.tensor([[0.2]]), LaplaceGlobalWidth([12, 128, 128], train_width=False)),
    ]:
        method.train()  # try train mode
        batch_x = torch.rand(size=(128, 12))  # dummy input
        pred_params = method(batch_x)  # try forward
        target = torch.rand(size=(128,))  # dummy target
        loss = method.loss(target, pred_params)  # try loss
        # try learning step
        optim = torch.optim.SGD(method.parameters(), lr=1e-3)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # see if the F value is correct
        if isinstance(method, LaplaceGlobalWidth):
            method.global_width.data = torch.tensor([0.5])
        CDFaty = method.get_F_at_y(torch.tensor([[0.7]]), pred_params=fixed_pred_params)
        assert torch.isclose(CDFaty, torch.tensor([[0.816]]), atol=1e-3)
        # see if the f value is correct
        logscoreaty = method.get_logscore_at_y(
            torch.tensor([[0.7]]), pred_params=fixed_pred_params
        )
        assert torch.isclose(logscoreaty, torch.tensor([[1.0]]), atol=1e-3)


def test_gaussian():
    torch.autograd.set_detect_anomaly(True)
    method = MixtureDensityNetwork([12, 128, 128], 2)
    fixed_pred_params = torch.tensor([[0.5, 0.5, 0, 2.0, 1.0, 0.5]]).expand(
        4, 6
    )  # batch size will be 4
    method.train()
    batch_x = torch.rand(size=(128, 12))
    pred_params = method(batch_x)
    target = torch.rand(size=(128,))
    loss = method.loss(target, pred_params)
    optim = torch.optim.SGD(method.parameters(), lr=1e-3)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # correct values come from wolframalpha
    # see if the F value is correct
    CDFaty = method.get_F_at_y(
        torch.tensor([-1, 0, 1, 2]), pred_params=fixed_pred_params
    )
    assert torch.allclose(
        CDFaty, torch.tensor([0.0793, 0.2500, 0.4320, 0.7386]), atol=1e-3
    )
    # see if the f value is correct
    logscoreaty = method.get_logscore_at_y(
        torch.tensor([-1, 0, 1, 2]), pred_params=fixed_pred_params
    )
    assert torch.allclose(
        logscoreaty, torch.tensor([2.1121, 1.6114, 1.7431, 0.8535]), atol=1e-3
    )


def test_categorical_cross_entropy():
    torch.autograd.set_detect_anomaly(True)

    # Assuming `MLP` and `CategoricalCrossEntropy` are defined as shown earlier
    # Setup model
    method = CategoricalCrossEntropy(
        [12, 128, 128], bin_borders=[0.0, 1.0, 2.0, 3.0, 4.0]
    )
    method.train()

    # Fixed predicted parameters for testing (batch size 4, B=4 bins)
    fixed_pred_params = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],  # distribution over bins
            [0.2, 0.2, 0.4, 0.2],
            [0.25, 0.25, 0.25, 0.25],
        ]
    ).expand(3, 4)

    # Create random batch_x for a forward pass
    batch_x = torch.rand(size=(128, 12))
    pred_params = method(batch_x)

    # Create random target values within the range of bin_borders
    target = torch.rand(size=(128,)) * 4  # since bin borders range from 0 to 4

    # Calculate loss
    loss = method.loss(target, pred_params)

    # Optimizer setup
    optim = torch.optim.SGD(method.parameters(), lr=1e-3)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Testing the CDF values
    CDFaty = method.get_F_at_y(
        torch.tensor([0.5, 1.5, 2.5]), pred_params=fixed_pred_params
    )
    assert torch.allclose(
        CDFaty, torch.tensor([0.05, 0.3, 0.625]), atol=1e-3
    ), f"CDF values are incorrect: {CDFaty}"

    # Testing the logscore values
    logscoreaty = method.get_logscore_at_y(
        torch.tensor([0.5, 1.5, 2.5]), pred_params=fixed_pred_params
    )
    assert torch.allclose(
        logscoreaty, torch.tensor([2.3026, 1.6094, 1.3863]), atol=1e-3
    ), f"Log score values are incorrect: {logscoreaty}"


def test_all():
    test_laplace()
    test_gaussian()
    test_categorical_cross_entropy()


if __name__ == "__main__":
    test_all()


def get_method(method_name):
    if method_name == "laplacescore":
        return LaplaceLogScore
    elif method_name == "laplacewb":
        return LaplaceGlobalWidth
    elif method_name == "mdn":
        return MixtureDensityNetwork
    elif method_name == "ce":
        return CategoricalCrossEntropy
    elif method_name == "pinball":
        return PinballLoss
    else:
        raise ValueError(f"Unknown method name {method_name}")
