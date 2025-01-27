import logging
import json
import os
import requests
import zipfile
import numpy as np

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
with open("config/kaggle.json", "r") as f:
    kaggle_api = json.load(f)


def download_flickr8k(save_path: str = "flickr-8k/raw/flickr-8k.zip"):
    """
    Downloads the Flickr8k dataset from Kaggle and saves it to the specified path.

    Args:
        save_path (str): The path to save the downloaded Flickr8k dataset zip file. Defaults to "flickr-8k/raw/flickr-8k.zip".

    Raises:
        FileNotFoundError: If the directory for the `save_path` does not exist and cannot be created.
        requests.exceptions.RequestException: If there is an error downloading the dataset from Kaggle.
    """
    # Ensure the directory for save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        logger.info(f"File already exists at {save_path}. Skipping download.")
        return

    dataset_url = (
        "https://www.kaggle.com/api/v1/datasets/download/dibyansudiptiman/flickr-8k"
    )
    headers = {"Authorization": f"Bearer {kaggle_api['key']}"}

    response = requests.get(dataset_url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        logger.info(f"Dataset downloaded successfully to {save_path}")
    else:
        logger.error(f"Error downloading dataset: {response.status_code}")


def unzip_and_organize_flickr8k(
    zip_path: str = "flickr-8k/raw/flickr-8k.zip",
    extract_to: str = "flickr-8k/intermediate",
):
    """
    Unzips the Flickr8k dataset zip file and organizes the extracted files into the specified directory.

    Args:
        zip_path (str): The path to the Flickr8k dataset zip file. Defaults to "flickr-8k/raw/flickr-8k.zip".
        extract_to (str): The path to the directory where the dataset should be extracted. Defaults to "flickr-8k/intermediate".

    Raises:
        FileNotFoundError: If the zip file does not exist at the specified `zip_path`.
    """
    if not os.path.exists(zip_path):
        logger.error(
            f"Zip file does not exist at {zip_path}, run download_flickr8k first."
        )
        return
    elif os.path.exists(os.path.join(extract_to, "Flickr8k.token.txt")):
        logger.info(
            f"Dataset already exists at {extract_to}. Skipping unzip and organize."
        )
        return

    if extract_to == "":
        extract_to = os.path.dirname(zip_path)
    else:
        os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        logger.info(f"Dataset unzipped successfully to {extract_to}")


def prepare_flickr8k_csv(data_path: str = "data/flickr-8k/intermediate"):
    """
    Prepares a CSV file containing the image captions from the Flickr8k dataset.

    The function reads the Flickr8k.token.txt file, which contains the caption IDs and their corresponding captions,
    and creates a pandas DataFrame with the data. The DataFrame is then saved to a CSV file in the "processed" subdirectory.

    Args:
        data_path (str): The path to the intermediate directory containing the Flickr8k dataset files.

    Raises:
        FileNotFoundError: If the Flickr8k.token.txt file does not exist in the specified data path.
    """
    token_file_path = os.path.join(data_path, "Flickr8k.token.txt")
    if not os.path.exists(token_file_path):
        logger.error(f"Token file does not exist at {token_file_path}.")
        return

    data = []
    with open(token_file_path, "r") as file:
        for line in file:
            caption_id, caption = line.strip().split("\t")
            data.append({"caption_id": caption_id, "caption": caption})

    df = pd.DataFrame(data)

    csv_path = os.path.join(
        os.path.dirname(data_path), "processed", "flickr8k_captions.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Captions CSV file created successfully at {csv_path}")
    return df


def prepare_expert_annotations(data_path: str = "data/flickr-8k/intermediate"):
    """
    Prepares a CSV file containing the expert annotations from the Flickr8k dataset.

    The function reads the ExpertAnnotations.txt file, which contains the image names,
    caption IDs, and expert judgments. It computes the average rating from the three
    expert judgments and creates a pandas DataFrame with the data. The DataFrame is
    then saved to a CSV file in the "processed" subdirectory of the data path.

    Args:
        data_path (str): The path to the intermediate directory containing the Flickr8k dataset files.
                         Defaults to "data/flickr-8k/intermediate".

    Raises:
        FileNotFoundError: If the ExpertAnnotations.txt file does not exist in the specified data path.
    """
    annotations_file_path = os.path.join(data_path, "ExpertAnnotations.txt")
    if not os.path.exists(annotations_file_path):
        logger.error(
            f"Expert annotations file does not exist at {annotations_file_path}."
        )
        return

    data = []
    with open(annotations_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                image, caption_id = parts[0], parts[1]
                ratings = [float(r) for r in parts[2:]]
                avg_rating = np.mean(ratings)
                data.append(
                    {"image": image, "caption_id": caption_id, "avg_rating": avg_rating}
                )

    df = pd.DataFrame(data)

    csv_path = os.path.join(
        os.path.dirname(data_path), "processed", "flickr8k_expert_annotations.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Expert annotations CSV file created successfully at {csv_path}")
    return df


def prepare_crowdflower_annotations(data_path: str = "data/flickr-8k/intermediate"):
    """
    Prepares a CSV file containing the CrowdFlower annotations from the Flickr8k dataset.

    The function reads the CrowdFlowerAnnotations.txt file, which contains the image names,
    caption IDs, and crowd judgments. It extracts the percentage of 'Yes' responses and
    creates a pandas DataFrame with the data. The DataFrame is then saved to a CSV file
    in the "processed" subdirectory of the data path.

    Args:
        data_path (str): The path to the intermediate directory containing the Flickr8k dataset files.
                         Defaults to "data/flickr-8k/intermediate".

    Raises:
        FileNotFoundError: If the CrowdFlowerAnnotations.txt file does not exist in the specified data path.
    """
    annotations_file_path = os.path.join(data_path, "CrowdFlowerAnnotations.txt")
    if not os.path.exists(annotations_file_path):
        logger.error(
            f"CrowdFlower annotations file does not exist at {annotations_file_path}."
        )
        return

    data = []
    with open(annotations_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                image, caption_id, percent_yes = parts[0], parts[1], float(parts[2])
                data.append(
                    {
                        "image": image,
                        "caption_id": caption_id,
                        "percent_yes": percent_yes,
                    }
                )

    df = pd.DataFrame(data)

    csv_path = os.path.join(
        os.path.dirname(data_path), "processed", "flickr8k_crowdflower_annotations.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"CrowdFlower annotations CSV file created successfully at {csv_path}")
    return df


def main():
    download_flickr8k("data/flickr-8k/raw/flickr-8k.zip")
    unzip_and_organize_flickr8k(
        "data/flickr-8k/raw/flickr-8k.zip", "data/flickr-8k/intermediate"
    )
    prepare_flickr8k_csv("data/flickr-8k/intermediate")
    prepare_expert_annotations("data/flickr-8k/intermediate")
    prepare_crowdflower_annotations("data/flickr-8k/intermediate")


if __name__ == "__main__":
    main()
