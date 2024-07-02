import numpy as np
from scipy.spatial.distance import mahalanobis

# Global variable to store the inverse covariance matrix
inv_cov = None

def compute_inv_cov(embeddings):
    global inv_cov
    if inv_cov is None:
        # Compute the covariance matrix
        cov = np.cov(embeddings.T)
        # Add a small constant to the diagonal to ensure positive definiteness
        cov += np.eye(cov.shape[0]) * 1e-6
        # Compute the inverse of the covariance matrix
        inv_cov = np.linalg.inv(cov)
    return inv_cov

def mahalanobis_distance(x, y, inv_cov):
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
    return mahalanobis(x, y, inv_cov)