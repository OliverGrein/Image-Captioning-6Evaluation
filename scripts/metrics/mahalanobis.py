import numpy as np
import numpy.typing as npt

# Global variable to store the inverse covariance matrix
inv_cov = None

def compute_inv_cov(embeddings):
    """
    This function computes the inverse covariance matrix for a set of embeddings.

    Args:
        embeddings (numpy.ndarray): The set of embeddings.

    Returns:
        numpy.ndarray: The inverse covariance matrix.
    """
    global inv_cov
    if inv_cov is None:
        cov = np.cov(embeddings.T)
        # Add a small constant to the diagonal to ensure positive definiteness
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov)
    return inv_cov

def mahalanobis_distance(reference: npt.NDArray, candidate: npt.NDArray, inv_cov: npt.NDArray):
    """
    This function measures the Mahalanobis distance between two n-dimensional vectors.

    Args:
        reference (numpy.ndarray): The reference vector.
        candidate (numpy.ndarray): The candidate vector.
        inv_cov (numpy.ndarray): The inverse of the covariance matrix.

    Returns:
        float: The Mahalanobis distance between the two vectors.

    Raises:
        ValueError: If the input vectors have different shapes.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Input vectors must have the same shape.")
    
    delta = reference - candidate
    m = np.dot(np.dot(delta, inv_cov), delta)
    return np.sqrt(m)