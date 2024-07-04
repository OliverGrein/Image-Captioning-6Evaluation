import numpy as np 
import numpy.typing as npt


def euclidian_distance(reference: npt.NDArray, candidate: npt.NDArray) -> float:
    """
    This function measures the Euclidian distance between two n-dimensional vectors.

    Args:
        reference (numpy.ndarray): The reference vector.
        candidate (numpy.ndarray): The candidate vector.

    Returns:
        float: The Euclidian distance between the two vectors.
    
    Raises:
        ValueError: If the input vectors have different shapes.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Input vectors must have the same shape.")

    euclidian_distance = np.linalg.norm(reference - candidate)
    
    return euclidian_distance