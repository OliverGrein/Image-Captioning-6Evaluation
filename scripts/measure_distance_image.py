import numpy as np
import os
import sys
import csv
import pandas as pd
import re
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path

from metrics.chebyshev import chebyshev_distance
from metrics.cosine import cosine_similarity 
from metrics.euclidian import euclidian_distance
from metrics.minkowski import minkowski_distance
from metrics.mahalanobis import mahalanobis_distance

CHUNK = 0
CHUNK_SIZE = 500
NUM_CHUNKS = 5

metrics = {
    "chebyshev": chebyshev_distance,
    "cosine": cosine_similarity,
    "euclidian": euclidian_distance,
    "minkowski": minkowski_distance,
    "mahalanobis": mahalanobis_distance
}


data_path = (Path(__file__).resolve().parent.parent / "data")
dataset_paths = {
    "cohere": Path(data_path / "cohere"),
    "openai": Path(data_path / "openai"),
    "vertexai": Path(data_path / "vertexai"),
    "voyageai": Path(data_path / "voyageai")
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

                distance = metrics[metric](reference, candidate, p) if metric == "minkowski" else metrics[metric](reference, candidate)

                values.append({"reference": images_df.iloc[i]['image'], "candidate": images_df.iloc[j]['image'], "distance": distance})
        
        out_df = pd.DataFrame(values, columns=["reference", "candidate", "distance"])
        print(out_df)

        out_df.to_csv(Path(data_path / f"{dataset.replace('/', '_')}_{metric}_{image.replace('.', '_')}_distances.csv").as_posix(), index=False)

    else:
        raise NotImplementedError("Calculating the distance for all images in the dataset is not yet implemented.")


import numpy as np
import pandas as pd
import re
from multiprocessing import Pool, cpu_count
from functools import partial

def process_image_embedding(image_embedding, target_shape):
    n_sections = target_shape[0] // image_embedding.shape[1]
    
    section_means = []
    for i in range(n_sections):
        start = i * image_embedding.shape[0] // n_sections
        end = (i + 1) * image_embedding.shape[0] // n_sections
        section_mean = np.mean(image_embedding[start:end], axis=0)
        section_means.append(section_mean)
    
    processed_embedding = np.concatenate(section_means)
    
    if processed_embedding.shape[0] < target_shape[0]:
        repetitions = int(np.ceil(target_shape[0] / processed_embedding.shape[0]))
        processed_embedding = np.tile(processed_embedding, repetitions)[:target_shape[0]]
    
    return processed_embedding

def process_image(base_image, df, image_embeddings, metrics, p):
    images_df = df[df['image'].apply(lambda x: re.sub(r'#\d+$', '', x)) == base_image].copy()
    
    if len(images_df) < 2:
        print(f"Skipping image {base_image} as it has only one embedding. At least 2 are needed to compare distances.")
        return []

    # Get the image embedding
    if image_embeddings is not None:
        #TODO: Select the image embeddings. 
        image_embedding_sub_df = image_embeddings[image_embeddings['image'].apply(lambda x: re.sub(r'#\d+$', '', x)) == base_image].copy()
    else:
        print(f"Image embedding not found for {base_image}. Skipping.")
        return []

    values = []
    for metric_name, metric_func in metrics.items():
        for i in range(len(images_df) - 1):
            for j in range(i+1, len(images_df)):
                reference = np.array(eval(images_df.iloc[i]['embedding']))
                candidate = np.array(eval(images_df.iloc[j]['embedding']))
                image_embedding =  image_embedding_sub_df.iloc[j]['embedding']
                # Preprocess the image embedding
                image_embedding_processed = process_image_embedding(image_embedding, reference.shape)
                
                if metric_name == "minkowski":
                    image_distance_reference = metric_func(image_embedding_processed, reference, p)
                    image_distance_candidate = metric_func(image_embedding_processed, candidate, p)
                else:
                    image_distance_reference = metric_func(image_embedding_processed, reference)
                    image_distance_candidate = metric_func(image_embedding_processed, candidate)

                values.append({
                    "metric": metric_name,
                    "image": base_image,
                    "reference": f"{base_image}_embedding_{i}",
                    "candidate": f"{base_image}_embedding_{j}",
                    "image_distance_reference": image_distance_reference,
                    "image_distance_candidate": image_distance_candidate
                })

    print(f"Processed image: {base_image}")
    return values

def measure_distance_image(dataset: str, image_embeddings: pd.DataFrame = None, p: float = 2):
    """
    This function measures the distance for all reference and candidate embeddings for all images in the specified embedding dataset.
    Data is stored in the Image-Captioning-6Evaluation/data/ directory.

    Args:
        dataset (str): The dataset to use (e.g. "cohere/flickr8k").
        image_embeddings (pd.DataFrame, optional): DataFrame containing image embeddings. Index should be image names, and values should be embeddings.
        p (float): The order of the Minkowski distance. Default is 2 (Euclidean distance). Only used if metric is "minkowski".
    """
    available_datasets = get_available_embedding_datasets()
    if dataset not in available_datasets.keys():
        print(f"Dataset {dataset} not found.")
        return

    df = pd.read_csv(available_datasets[dataset])
    df['image'] = df['image'].str[:-2]
    df = df[df['image'].str.endswith('.jpg')]
    df = df.iloc[CHUNK * CHUNK_SIZE * NUM_CHUNKS: (CHUNK + 1) * NUM_CHUNKS * CHUNK_SIZE]

    # Get unique base image names
    unique_base_images = df['image'].apply(lambda x: re.sub(r'#\d+$', '', x)).unique()

    # # Set up multiprocessing
    # num_processes = cpu_count()
    # pool = Pool(processes=num_processes)

    # # Partial function to pass static arguments
    # process_image_partial = partial(process_image, df=df, image_embeddings=image_embeddings, metrics=metrics, p=p)

    # # Process images in parallel
    # all_values = pool.map(process_image_partial, unique_base_images)

    # # Flatten the list of results
    # all_values = [item for sublist in all_values for item in sublist]

    # pool.close()
    # pool.join()

    all_values = []
    for base_image in unique_base_images:
        chunk_values = process_image(base_image, df, image_embeddings, metrics, p)
        all_values.extend(chunk_values)

    out_df = pd.DataFrame(all_values, columns=["metric", "image", "reference", "candidate", "image_distance_reference", "image_distance_candidate"])
    print(out_df)
    out_df.to_csv(f"{dataset}_all_images_distances_{CHUNK}.csv", index=False)

def load_image_embeddings(directory):
    concatenated_array = None

    # List to store filenames
    filenames = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .npy file
        if filename.endswith('.npy'):
            filenames.append(filename)
    
    # Sort filenames numerically based on the numerical part
    filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Limit the number of files to load
    filenames = filenames[CHUNK * NUM_CHUNKS: (CHUNK + 1) * NUM_CHUNKS]
    
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        # Load the numpy array from the file
        array = np.load(file_path)
        print(f"loaded {file_path}")
        # Append the array to the concatenated_array
        if concatenated_array is None:
            concatenated_array = array
        else:
            concatenated_array = np.concatenate((concatenated_array, array), axis=0)
    
    return concatenated_array

def create_image_embeddings_df(image_names, embeddings):
    if len(image_names) != embeddings.shape[0]:
        raise ValueError("The number of image names does not match the number of embeddings.")
    
    df = pd.DataFrame({
        'image': image_names,
        'embedding': list(embeddings)
    })
    return df

if __name__ == "__main__":
    # Path to the captions.csv file
    captions_path = (Path(__file__).resolve().parent.parent / "data" / "flickr-8k" / "processed" / "flickr8k_captions.csv")
    captions_df = pd.read_csv(captions_path)

    while CHUNK < 17:
        # Get unique image names from the captions.csv
        unique_image_names = captions_df['image'].iloc[CHUNK * CHUNK_SIZE * NUM_CHUNKS: (CHUNK + 1) * NUM_CHUNKS * CHUNK_SIZE]
        print(f"Chunk from {CHUNK * CHUNK_SIZE * NUM_CHUNKS} to {(CHUNK + 1) * NUM_CHUNKS * CHUNK_SIZE}")

        # Path to the directory containing .npy files
        data_path = (Path(__file__).resolve().parent.parent / "data" / "flickr-8k" / "image_emb")
        flickr_image_embeddings = load_image_embeddings(data_path)

        # Create DataFrame using the image names and loaded embeddings
        image_embeddings_df = create_image_embeddings_df(unique_image_names, flickr_image_embeddings)

        # Measure distances using the created DataFrame
        measure_distance_image(dataset="cohere/flickr8k", image_embeddings=image_embeddings_df)

        CHUNK = CHUNK + 1