import numpy as np
import numpy.typing as npt


def minkowski_distance(reference: npt.NDArray, candidate: npt.NDArray, p: float = 2) -> float:
    """
    This function measures the Minkowski distance between two n-dimensional vectors.

    Args:
        reference (numpy.ndarray): The reference vector.
        candidate (numpy.ndarray): The candidate vector.
        p (float): The order of the Minkowski distance. Default is 2 (Euclidean distance).

    Returns:
        float: The Minkowski distance between the two vectors.
    
    Raises:
        ValueError: If the input vectors have different shapes.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Input vectors must have the same shape.")

    minkowski_distance = np.power(np.sum(np.abs(reference - candidate) ** p), 1/p)
    
    return minkowski_distance