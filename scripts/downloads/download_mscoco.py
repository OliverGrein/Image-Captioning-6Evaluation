import logging
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
import zipfile

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, save_path):
    """
    Downloads a file from the specified URL and saves it to the specified path.

    Args:
        url (str): The URL to download the file from.
        save_path (str): The path to save the downloaded file.

    Raises:
        requests.exceptions.RequestException: If there is an error downloading the file.
    """
    if os.path.exists(save_path):
        logger.info(f"File {save_path} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        logger.info(f"Downloaded {save_path}")
    else:
        logger.error(f"Error downloading file: {response.status_code}")


def unzip_file(zip_path: str, extract_to: str):
    """
    Unzips a file to the specified directory.

    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory to extract the contents to.

    Raises:
        zipfile.BadZipFile: If the file is not a zip file or it is corrupted.
    """
    if not os.path.exists(zip_path):
        logger.error(f"File {zip_path} does not exist.")
        return

    if os.path.exists(extract_to):
        logger.info(f"Directory {extract_to} already exists. Skipping extraction.")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        logger.info(f"Extracted {zip_path} to {extract_to}")


def parse_annotations(caption_path: str, df_path: str):
    """
    Parses the MS-COCO annotations JSON file and saves the annotations to a CSV file.

    Args:
        caption_path (str): The path to the MS-COCO annotations JSON file.
        df_path (str): The path to save the annotations as a CSV file.

    Raises:
        FileNotFoundError: If the annotations JSON file does not exist.
    """
    if not os.path.exists(caption_path):
        logger.error(f"File {caption_path} does not exist.")
        return

    if os.path.exists(df_path):
        logger.info(f"Annotations already saved to {df_path}. Skipping parsing.")
        return
    os.makedirs(os.path.dirname(df_path), exist_ok=True)

    with open(caption_path, "r") as f:
        json_blob = json.load(f)

    annotations = json_blob["annotations"]

    df = pd.DataFrame(annotations)
    df = df.sort_values(by="image_id")
    df.to_csv(df_path, index=False)
    logger.info(f"Annotations saved to {df_path}")


def main():
    # URLs for MS-COCO dataset
    urls = [
        # "http://images.cocodataset.org/zips/train2017.zip",
        # "http://images.cocodataset.org/zips/val2017.zip",
        # "http://images.cocodataset.org/zips/test2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    ]

    # Directory to save the dataset
    save_dir = "data/mscoco/raw"
    os.makedirs(save_dir, exist_ok=True)

    # Download each file in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                download_file, url, os.path.join(save_dir, url.split("/")[-1])
            )
            for url in urls
        ]
        for future in tqdm(futures, desc="Downloading files"):
            future.result()

    intermediate_dir = os.path.join(os.path.dirname(save_dir), "intermediate")
    unzip_file(os.path.join(save_dir, "annotations_trainval2017.zip"), intermediate_dir)

    processed_dir = os.path.join(os.path.dirname(save_dir), "processed", "captions.csv")
    parse_annotations(
        "data/mscoco/intermediate/annotations/captions_val2017.json", processed_dir
    )


if __name__ == "__main__":
    main()
