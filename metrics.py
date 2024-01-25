# gpt4 says:
import numpy as np
from scipy.integrate import quad


def crps_single_prediction(predicted_cdf, ground_truth, lower_bound, upper_bound):
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
        integrand, lower_bound, upper_bound, epsabs=1e-04, epsrel=1e-04, limit=5
    )
    return crps


def crps_batch_prediction(get_fn_cdf, get_bounds_fn, predictions, ground_truths):
    """
    Compute the mean CRPS for a batch of predictions.

    :param get_cdf_func: A function that given a prediction returns the predicted cdf
    :param predictions: The predictions of the parameters of the cdfs
    :param ground_truths: A list of ground truth values corresponding to each prediction.
    :param lower_bound: Lower bound for integration.
    :param upper_bound: Upper bound for integration.
    :return: Mean CRPS for the given batch of predictions.
    """
    lower_bound, upper_bound = get_bounds_fn(predictions)
    predicted_cdfs = [get_fn_cdf(pred) for pred in predictions]
    crps_scores = [
        crps_single_prediction(cdf, gt, lower_bound, upper_bound)
        for cdf, gt in zip(predicted_cdfs, ground_truths)
    ]
    mean_crps = np.mean(crps_scores)
    return mean_crps
