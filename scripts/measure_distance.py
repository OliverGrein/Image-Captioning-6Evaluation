import numpy as np
import sys
import csv
import pandas as pd

from pathlib import Path

from metrics.chebyshev import chebyshev_distance
from metrics.cosine import cosine_similarity 
from metrics.euclidian import euclidian_distance
from metrics.minkowski import minkowski_distance


metrics = {
    "chebyshev": chebyshev_distance,
    "cosine": cosine_similarity,
    "euclidian": euclidian_distance,
    "minkowski": minkowski_distance
}


data_path = (Path(__file__).resolve().parent.parent / "data")
dataset_paths = {
    "cohere": Path(data_path / "cohere"),
    "openai": Path(data_path / "openai"),
    "vertexai": Path(data_path / "vertexai")
}


def get_available_embedding_datasets() -> dict:
    """
    This function checks the available datasets in the Image-Captioning-6Evaluation/data/ directory.

    Returns:
        dict: The available datasets.
    """
    available_datasets = {}
    for name, path in dataset_paths.items():
        if Path(path).exists():
            flickr8k = Path(path / "flickr8k_embeddings.csv")
            if flickr8k.is_file():
                available_datasets[f"{name}/flickr8k"] = flickr8k

            is_mscoco = Path(path / "mscoco_embeddings.csv")
            if is_mscoco.is_file():
                available_datasets[f"{name}/mscoco"] = is_mscoco

            is_pascal50s = Path(path / "pascal50s_embeddings.csv")
            if is_pascal50s.is_file():
                available_datasets[f"{name}/pascal50s"] = is_pascal50s

    if not available_datasets: # empty dict evaluates to False
        print("WARN: No datasets found. Try running `scripts/bootstrap.py` from the main directory first.")

    return available_datasets


def measure_distance(dataset: str, metric: str, image: str = None):
    """
    This function measures the distance for all reference and candidate embeddings for an image in one of the embedding datasets.
    Data is stored in the Image-Captioning-6Evaluation/data/ directory.
    
    If no image is specified, the function will measure the distance for all images in the dataset.

    Args:
        dataset (str): The dataset to use (e.g. "cohere/flickr8k").
        metric (str): The metric to use. ("chebushev", "cosine", or "euclidian").
        image (str): The image to use. (e.g. "1000268201_693b08cb0e.jpg")

    """
    available_datasets = get_available_embedding_datasets()
    if dataset not in available_datasets.keys():
        print(f"Dataset {dataset} not found.")
        return

    if metric not in metrics:
        print(f"Metric {metric} not found.")
        return

    df = pd.read_csv(available_datasets[dataset])

    if image:
        images_df = df[df['image'].str.contains(f'{image}')].copy() # Get all images that start with the given image name

        # Validation
        if images_df.empty:
            raise KeyError(f"Image {image} not found in dataset {dataset}.")
        if len(images_df) < 2:
            raise ValueError(f"Image {image} has only one embedding. At least 2 are needed to compare distances.")

        values = []

        # Get the reference embedding
        for i in range(len(images_df) - 1):
            for j in range(i+1, len(images_df)):
                reference = np.array(eval(images_df.iloc[i]['embedding']))
                candidate = np.array(eval(images_df.iloc[j]['embedding']))        

                distance = metrics[metric](reference, candidate)

                values.append({"reference": images_df.iloc[i]['image'], "candidate": images_df.iloc[j]['image'], "distance": distance})
        
        out_df = pd.DataFrame(values, columns=["reference", "candidate", "distance"])
        print(out_df)

        out_df.to_csv(Path(data_path / f"{dataset.replace("/", "_")}_{metric}_{image.replace(".", "_")}_distances.csv").as_posix(), index=False)

    else:
        raise NotImplementedError("Calculating the distance for all images in the dataset is not yet implemented.")

if __name__ == "__main__":
    measure_distance(dataset="openai/flickr8k", metric="cosine", image="1000268201_693b08cb0e.jpg")
