import torch

from base_methods import BaseProbabilisticMethod

class MeanValStd(BaseProbabilisticMethod):
    def get_mlp_output_dim(self):
        return 1  # predict the mean only

    def compute_extra_params(self, cal_preds, cal_targets):
        self.std = torch.std(cal_preds - cal_targets) # add the other value

    def predict_cdf(self, dist_param_prediction, point_to_evaluate):
        """Given the mean and the variance we compute the P(X<`point_to_evaluate`)."""
        mean = dist_param_prediction
        # evaluate point given mean and std
        dist = torch.distributions.Normal(mean, self.std)
        cdf = dist.cdf(torch.Tensor([point_to_evaluate]))
        return cdf.item()

    def loss_fn(self, y_pred, y_target):
        return torch.mean((y_pred - y_target) ** 2)
        

methods = {
    "MeanValStd": MeanValStd,
}

