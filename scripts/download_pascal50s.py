import logging
import requests
import os

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL of the PASCAL-50S dataset page
url = "https://hockenmaier.cs.illinois.edu/pascal-sentences/index.html"


def download_image(url, save_path):
    """
    Downloads an image from the specified URL and saves it to the specified path.

    Args:
        url (str): The URL of the image to download.
        save_path (str): The path to save the downloaded image.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        logger.error(f"Error downloading image: {response.status_code}")


def download_pascal50s(save_dir: str = "pascal50s"):
    """
    Downloads the PASCAL-50S dataset images and captions, and saves them to the specified directory.

    Args:
        save_dir (str): The directory to save the images and captions. Defaults to "pascal50s".
    """
    os.makedirs(save_dir, exist_ok=True)

    # Check if data already exists
    captions_file = os.path.join(save_dir, "captions.csv")
    if os.path.exists(captions_file):
        logger.info("Data already exists. Skipping download.")
        return

    # Fetch the webpage
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Error fetching webpage: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, "html.parser")

    data = []

    def process_image(img_tag):
        img_url = img_tag["src"]
        img_filename = os.path.join(save_dir, "images", os.path.basename(img_url))
        if not os.path.exists(img_filename):
            os.makedirs(os.path.dirname(img_filename), exist_ok=True)
        # Download the image
        download_image(
            f"https://hockenmaier.cs.illinois.edu/pascal-sentences/{img_url}",
            img_filename,
        )

        # Find the corresponding captions
        captions = img_tag.find_next("table").find_all("td")
        for caption in captions:
            data.append([os.path.basename(img_url), caption.text.strip()])

    img_tags = soup.find_all("img")
    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_image, img_tags),
                total=len(img_tags),
                desc="Downloading images and captions",
            )
        )

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["image", "caption"])
    captions_file = os.path.join(save_dir, "captions.csv")
    df.to_csv(captions_file, index=False)

    logger.info("Scraping complete!")


def main():
    download_pascal50s("data/pascal50s/raw")


if __name__ == "__main__":
    main()
