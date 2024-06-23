import numpy as np
import numpy.typing as npt


def chebyshev_distance(reference: npt.NDArray, candidate: npt.NDArray) -> float:
    """
    This function measures the Chebyshev distance (Also chessboard or L-Infinity distance) between two n-dimensional vectors.

    Args:
        reference (numpy.ndarray): The reference vector.
        candidate (numpy.ndarray): The candidate vector.

    Returns:
        float: The Chebyshev distance between the two vectors.
    
    Raises:
        ValueError: If the input vectors have different shapes.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Input vectors must have the same shape.")

    chebyshev_distance = 0.0
    for i in range(len(reference.shape)):
        val = np.abs(reference[i] - candidate[i])
        if (val > chebyshev_distance):
            chebyshev_distance = val

    return chebyshev_distance
