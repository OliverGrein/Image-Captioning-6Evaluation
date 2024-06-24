import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


def mahalanobis_distance(reference: npt.NDArray, candidate: npt.NDArray) -> float:
    """
    This function measures the Mahalanobis distance between two n-dimensional vectors.

    Args:
        reference (numpy.ndarray): The reference vector.
        candidate (numpy.ndarray): The candidate vector.

    Returns:
        float: The Mahalanobis distance between the two vectors.
    
    Raises:
        ValueError: If the input vectors have different shapes.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Input vectors must have the same shape.")

    # Combine the vectors into a single array
    X = np.vstack([reference, candidate])

    # Calculate the covariance matrix
    cov = np.cov(X.T)

    # Calculate the mean of the reference vector
    mean = np.mean(reference)

    # Mahalanobis distance
    mahalanobis_dist = multivariate_normal.mahalanobis(candidate, mean, cov)
    
    return mahalanobis_dist