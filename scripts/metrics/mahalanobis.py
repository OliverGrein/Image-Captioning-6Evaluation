import numpy as np
import numpy.typing as npt


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

    # Ensure the input arrays are 1D
    reference = reference.flatten()
    candidate = candidate.flatten()

    # Calculate the difference between candidate and reference
    diff = candidate - reference

    # Use the variance of each feature instead of covariance matrix
    var = np.var(np.vstack([reference, candidate]), axis=0)
    
    # Replace zero variances with a small positive number to avoid division by zero
    var = np.where(var == 0, 1e-8, var)

    # Mahalanobis distance
    mahalanobis_dist = np.sqrt(np.sum(diff**2 / var))
    
    return mahalanobis_dist