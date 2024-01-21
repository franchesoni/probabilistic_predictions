

# gpt4 says:
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
        return (predicted_cdf(x) - 1*(x >= ground_truth))**2

    crps, _ = quad(integrand, lower_bound, upper_bound)
    return crps
