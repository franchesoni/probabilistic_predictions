import numpy as np

def CRPS(predictions, targets):
    """
    Computes the CRPS score.
    `predictions` is a tuple with `bins` and `probs`.
    `targets` is an array of floats.
    """
    assert predictions.shape[0] == targets.shape[0]

def single_CRPS(bins, probs, target):
    assert sorted(bins) == bins
    assert len(bins) == len(probs) + 1
    # find the CDF

def get_CDF(bins, probs):
    assert sorted(bins) == bins
    assert len(bins) == len(probs) + 1
    """Gives the probability that a value is less than or equal to the bin."""
    CDF = np.zeros(len(bins))  # before, bin1 start, bin2 start, ..., bin N start, bin N end, after
    for i in range(1, len(bins)):
        CDF[i] = CDF[i - 1] + probs[i - 1]
    CDF[-1] = 1
    return CDF