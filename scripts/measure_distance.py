import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from metrics.chebyshev import chebyshev_distance
from metrics.cosine import cosine_similarity
from metrics.euclidian import euclidian_distance
from metrics.minkowski import minkowski_distance
from metrics.mahalanobis import mahalanobis_distance, compute_inv_cov


metrics = {
    "chebyshev": chebyshev_distance,
    "cosine": cosine_similarity,
    "euclidian": euclidian_distance,
    "minkowski": minkowski_distance,
    "mahalanobis": mahalanobis_distance,
}


data_path = Path(__file__).resolve().parent.parent / "data"
dataset_paths = {
    "cohere": Path(data_path / "cohere"),
    "openai": Path(data_path / "openai"),
    "vertexai": Path(data_path / "vertexai"),
    "voyageai": Path(data_path / "voyageai"),
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

    if not available_datasets:  # empty dict evaluates to False
        print(
            "WARN: No datasets found. Try running `scripts/bootstrap.py` from the main directory first."
        )

    return available_datasets


def measure_distance(dataset: str, metric: str, image: str = None, p: float = 2):
    """
    This function measures the distance for all reference and candidate embeddings for an image in one of the embedding datasets.
    Data is stored in the Image-Captioning-6Evaluation/data/ directory.

    If no image is specified, the function will measure the distance for all images in the dataset.

    Args:
        dataset (str): The dataset to use (e.g. "cohere/flickr8k").
        metric (str): The metric to use. ("chebyshev", "cosine", "euclidian", "minkowski", or "mahalanobis").
        image (str): The image to use. (e.g. "1000268201_693b08cb0e.jpg")
        p (float): The order of the Minkowski distance. Default is 2 (Euclidean distance). Only used if metric is "minkowski".

    """
    available_datasets = get_available_embedding_datasets()
    if dataset not in available_datasets.keys():
        print(f"Dataset {dataset} not found.")
        return

    if metric not in metrics:
        print(f"Metric {metric} not found.")
        return

    df = pd.read_csv(available_datasets[dataset])

    # Compute inverse covariance matrix for all embeddings
    all_embeddings = np.array([eval(emb) for emb in df["embedding"]])
    inv_cov = compute_inv_cov(all_embeddings)

    if image:
        images_df = df[
            df["image"].str.contains(f"{image}")
        ].copy()  # Get all images that start with the given image name

        # Validation
        if images_df.empty:
            raise KeyError(f"Image {image} not found in dataset {dataset}.")
        if len(images_df) < 2:
            raise ValueError(
                f"Image {image} has only one embedding. At least 2 are needed to compare distances."
            )

        values = []

        # Get the reference embedding
        for i in range(len(images_df) - 1):
            for j in range(i + 1, len(images_df)):
                reference = np.array(eval(images_df.iloc[i]["embedding"]))
                candidate = np.array(eval(images_df.iloc[j]["embedding"]))

                if metric == "minkowski":
                    distance = metrics[metric](reference, candidate, p)
                elif metric == "mahalanobis":
                    distance = metrics[metric](reference, candidate, inv_cov)
                else:
                    distance = metrics[metric](reference, candidate)

                values.append(
                    {
                        "reference": images_df.iloc[i]["image"],
                        "candidate": images_df.iloc[j]["image"],
                        "distance": distance,
                    }
                )

        out_df = pd.DataFrame(values, columns=["reference", "candidate", "distance"])
        print(out_df)

        out_df.to_csv(
            Path(
                data_path
                / f"{dataset.replace('/', '_')}_{metric}_{image.replace('.', '_')}_distances.csv"
            ).as_posix(),
            index=False,
        )

    else:
        raise NotImplementedError(
            "Calculating the distance for all images in the dataset is not yet implemented."
        )


def compute_expert_distances(
    expert_embeddings: str = "data/flickr-8k/processed/flickr8k_expert_annotations.csv",
    results_dir: str = "results",
):
    """
    Compute distances between expert-annotated captions and ground truth embeddings for multiple models and metrics.

    This function processes expert annotations and computes distances between the annotated captions
    and the corresponding ground truth embeddings for various embedding models and distance metrics.
    The results are saved as CSV files for each model-metric combination.

    Args:
        expert_embeddings (str): Path to the CSV file containing expert annotations.
            Default is "data/flickr-8k/processed/flickr8k_expert_annotations.csv".

    The function performs the following steps:
    1. Loads expert annotations and available embedding datasets.
    2. For each model and metric combination:
        a. Computes distances between expert-annotated captions and ground truth embeddings.
        b. Calculates average distances for each image-caption pair.
        c. Saves the results to a CSV file.

    The results are saved in the "data/results/" directory with filenames in the format:
    "{model}_{metric}_expert_distances.csv"

    Note:
    - This function assumes that the necessary embedding datasets are available and accessible.
    - It uses the global 'metrics' dictionary to access different distance metric functions.
    - Progress is displayed using tqdm progress bars.
    """
    expert_df = pd.read_csv(expert_embeddings)

    datasets = get_available_embedding_datasets()
    models = {
        dataset.split("/")[0]: path
        for dataset, path in datasets.items()
        if "flickr8k" in dataset
    }

    total_iterations = len(models) * len(metrics)
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for model in models:
            ground_truth_df = pd.read_csv(models[model])
            if model == "cohere":  #!TODO: delete once run again
                continue

            for metric in metrics:
                if metric == "mahalanobis":  # tbd on how to run manhalobis
                    continue

                results = []
                for _, row in tqdm(
                    expert_df.iterrows(),
                    total=len(expert_df),
                    desc=f"{model} - {metric}",
                ):
                    image = row["image"]
                    caption_id = row["caption_id"]
                    avg_rating = row["avg_rating"]

                    ground_truth_embeddings = ground_truth_df[
                        ground_truth_df["image"].apply(lambda x: x.split("#")[0])
                        == image
                    ]["embedding"].tolist()
                    ground_truth_embeddings = [
                        np.array(eval(emb)) for emb in ground_truth_embeddings
                    ]

                    candidate_embedding = ground_truth_df[
                        ground_truth_df["image"] == caption_id
                    ]["embedding"].values[0]
                    candidate_embedding = np.array(eval(candidate_embedding))

                    distances = [
                        metrics[metric](gt_emb, candidate_embedding)
                        for gt_emb in ground_truth_embeddings
                    ]
                    avg_distance = np.mean(distances)

                    results.append(
                        {
                            "image": image,
                            "caption_id": caption_id,
                            "avg_rating": avg_rating,
                            "avg_distance": avg_distance,
                        }
                    )

                results_df = pd.DataFrame(results)

                # Ensure the results directory exists
                results_dir = Path(f"{results_dir}/{model}")
                results_dir.mkdir(parents=True, exist_ok=True)

                results_df.to_csv(
                    results_dir / f"{model}_{metric}_expert_distances.csv", index=False
                )
                print(f"Saved results for {model} using {metric} metric")

                pbar.update(1)


if __name__ == "__main__":
    # measure_distance(
    #     dataset="openai/flickr8k", metric="cosine", image="1000268201_693b08cb0e.jpg"
    # )
    compute_expert_distances()
