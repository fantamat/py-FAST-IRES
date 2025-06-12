"""
ARfit: Quantiles of Student's t distribution.

This module implements the computation of quantiles of Student's t distribution.
"""

from scipy import stats

def tquant(dof, p):
    """
    Quantiles of Student's t distribution.
    
    Parameters
    ----------
    dof : float or array_like
        Degrees of freedom.
    p : float or array_like
        Probability values between 0 and 1.
    
    Returns
    -------
    q : float or ndarray
        Quantiles of the Student's t distribution with dof degrees of freedom,
        such that the probability of a value less than q is p.
    
    Notes
    -----
    This function computes quantiles of the Student's t distribution with dof
    degrees of freedom. For a probability p, the quantile q is defined by:
    P(t < q) = p where t is a random variable following Student's t distribution.
    
    This is required by ARCONF and ARMODE in the construction of confidence intervals.
    """
    return stats.t.ppf(p, dof)
