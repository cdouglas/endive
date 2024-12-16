import math
import numpy as np

def truncated_zipf_pmf(n, s):
    """
    Compute the truncated Zipf PMF for ranks 1 through n with exponent s.
    Returns a probability vector of length n.
    (OpenAI ChatGPT o1)
    """
    ranks = np.arange(1, n+1)
    weights = 1.0 / (ranks ** s)
    pmf = weights / weights.sum()
    return pmf

def lognormal_params_from_mean_and_sigma(mean_runtime_ms: float, sigma: float) -> (float, float):
    """
    Given the desired average (mean) runtime of a lognormal distribution and
    the chosen sigma (std. dev.) of the underlying normal distribution,
    compute the mu parameter for the underlying normal distribution.

    Parameters
    ----------
    mean_runtime_ms : float
        The desired mean of the lognormal distribution (e.g., 10,000 ms for 10 seconds).
    sigma : float
        The standard deviation of the underlying normal distribution that
        generates the lognormal distribution.

    Returns
    -------
    mu : float
        The mu parameter of the underlying normal distribution.
    sigma : float
        The sigma parameter of the underlying normal distribution (unchanged).
    """
    mu = math.log(mean_runtime_ms) - (sigma ** 2 / 2.0)
    return mu, sigma

