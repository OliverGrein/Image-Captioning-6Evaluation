import numpy as np
import numpy.typing as npt


def cosine_similarity(reference: npt.NDArray, candidate: npt.NDArray) -> float:
    """
    This function measures the cosine similarity between two n-dimensional vectors.

    Args:
        reference (numpy.ndarray): The reference vector.
        candidate (numpy.ndarray): The candidate vector.

    Returns:
        float: The cosine similarity between the two vectors.
    
    Raises:
        ValueError: If the input vectors have different shapes.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Input vectors must have the same shape.")

    dot_product = np.dot(reference, candidate)
    norm_reference = np.linalg.norm(reference)
    norm_candidate = np.linalg.norm(candidate)
    
    cosine_similarity = dot_product / (norm_reference * norm_candidate)
    
    return cosine_similarity