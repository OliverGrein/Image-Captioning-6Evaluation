import logging
import json
import os
import requests
import zipfile

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
with open("config/kaggle.json", "r") as f:
    kaggle_api = json.load(f)


def download_flickr8k(save_path: str = "flickr-8k/raw/flickr-8k.zip"):
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


def prepare_flickr8k_csv(data_path: str = "flickr-8k/intermediate"):
    token_file_path = os.path.join(data_path, "Flickr8k.token.txt")
    if not os.path.exists(token_file_path):
        logger.error(f"Token file does not exist at {token_file_path}.")
        return

    data = []
    with open(token_file_path, "r") as file:
        for line in file:
            image, caption = line.strip().split("\t")
            data.append({"image": image, "caption": caption})

    df = pd.DataFrame(data)

    csv_path = os.path.join(
        os.path.dirname(data_path), "processed/flickr8k_captions.csv"
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df.to_csv(csv_path, index=False)
    logger.info(f"CSV file created successfully at {csv_path}")


def main():
    download_flickr8k("data/flickr-8k/raw/flickr-8k.zip")
    unzip_and_organize_flickr8k(
        "data/flickr-8k/raw/flickr-8k.zip", "data/flickr-8k/intermediate"
    )
    prepare_flickr8k_csv("data/flickr-8k/intermediate")


if __name__ == "__main__":
    main()
