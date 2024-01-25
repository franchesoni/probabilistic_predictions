import torch

from base_methods import BaseProbabilisticMethod


class MeanValStd(BaseProbabilisticMethod):
    def __init__(self):
        self.std = 1

    @staticmethod
    def get_mlp_output_dim():
        return 1  # predict the mean only

    def compute_extra_params(self, cal_preds, cal_targets):
        self.std = torch.std(cal_preds - cal_targets)  # add the other value

    def predict_cdf(self, dist_param_prediction, point_to_evaluate):
        """Given the mean and the variance we compute the P(X<`point_to_evaluate`)."""
        return self.get_cdf_func(dist_param_prediction)(point_to_evaluate)

    def get_cdf_func(self, dist_param_prediction):
        mean = dist_param_prediction
        # evaluate point given mean and std
        dist = torch.distributions.Normal(mean, self.std)

        def cdf_func(point_to_evaluate):
            return dist.cdf(torch.Tensor([point_to_evaluate])).item()

        return cdf_func

    def get_bounds(self, predictions):
        lower_bound = predictions.min() - 4 * self.std
        upper_bound = predictions.max() + 4 * self.std
        return lower_bound, upper_bound

    @staticmethod
    def loss_fn(y_pred, y_target):
        return torch.mean((y_pred - y_target) ** 2)


class MeanStd(BaseProbabilisticMethod):
    @staticmethod
    def get_mlp_output_dim():
        return 2  # predict mean and std

    def compute_extra_params(self, cal_preds, cal_targets):
        pass

    @staticmethod
    def predict_cdf(dist_param_prediction, point_to_evaluate):
        """Given the mean and the variance we compute the P(X<`point_to_evaluate`)."""
        return MeanStd.get_cdf_func(dist_param_prediction)(point_to_evaluate)

    @staticmethod
    def get_cdf_func(dist_param_prediction):
        mean, var = dist_param_prediction
        var = torch.nn.functional.softplus(var)
        std = torch.sqrt(var)
        # evaluate point given mean and std
        dist = torch.distributions.Normal(mean, std)

        def cdf_func(point_to_evaluate):
            return dist.cdf(torch.Tensor([point_to_evaluate])).item()

        return cdf_func

    @staticmethod
    def get_bounds(predictions):
        mean, var = predictions[:, 0], torch.nn.functional.softplus(predictions[:, 1])
        std = torch.sqrt(var)
        lower_bound = (mean - 4 * std).min()
        upper_bound = (mean + 4 * std).max()
        return lower_bound, upper_bound

    @staticmethod
    def loss_fn(y_pred, y_target):
        mean, var = y_pred[:, 0], torch.nn.functional.softplus(y_pred[:, 1])
        return torch.nn.GaussianNLLLoss()(mean, y_target, var)


methods = {
    "MeanValStd": MeanValStd,
    "MeanStd": MeanStd,
}
