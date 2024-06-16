import pandas as pd
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

def get_openai_embeddings(model, text):
    url = "https://api.openai.com/v1/embeddings"
    api_key = os.getenv("OPENAI_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "input": text
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        embeddings = [item["embedding"] for item in response.json().get("data", [])]
        return embeddings
    else:
        raise Exception(f"Failed to get embeddings: {response.status_code}, {response.text}")

def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    model = "text-embedding-3-small"
    chunk_size = 2048

    flickr8k_captions = pd.read_csv("Image-Captioning-6Evaluation/data/flickr-8k/processed/flickr8k_captions.csv")
    mscoco_captions = pd.read_csv("Image-Captioning-6Evaluation/data/mscoco/processed/captions.csv")
    pascal50s_captions = pd.read("Image-Captioning-6Evaluation/data/pascal50s/raw/captions.csv")

    # Split captions into chunks that the APi can handle
    caption_chunks_flickr8k = list(split_into_chunks(flickr8k_captions['caption'].tolist(), chunk_size))
    caption_chunks_mscoco = list(split_into_chunks(mscoco_captions['caption'].tolist(), chunk_size))
    caption_chunks_pascal50s = list(split_into_chunks(pascal50s_captions['caption'].tolist(), chunk_size))

    # Get embeddings for each chunk
    flickr8k_embeddings = []
    for chunk in tqdm(caption_chunks_flickr8k, desc="Embedding chunks"):
        embeddings = get_openai_embeddings(model=model, text=chunk)
        flickr8k_embeddings.extend(embeddings)

    mscoco_embeddings = []
    for chunk in tqdm(caption_chunks_mscoco, desc="Embedding chunks"):
        embeddings = get_openai_embeddings(model=model, text=chunk)
        mscoco_embeddings.extend(embeddings)

    pascal50s_embeddings = []
    for chunk in tqdm(caption_chunks_pascal50s, desc="Embedding chunks"):
        embeddings = get_openai_embeddings(model=model, text=chunk)
        pascal50s_embeddings.extend(embeddings)

    flickr8k_captions["embedding"] = flickr8k_embeddings
    flickr8k_captions.to_csv("Image-Captioning-6Evaluation/data/flickr-8k/flickr8k_embeddings.csv")

    mscoco_captions["embedding"] = mscoco_embeddings
    mscoco_captions.to_csv("Image-Captioning-6Evaluation/data/mscoco/mscoco_embeddings.csv")

    pascal50s_captions["embedding"] = pascal50s_embeddings
    pascal50s_captions.to_csv("Image-Captioning-6Evaluation/data/pascal50s/pascal50s_embeddings.csv")

    print("Done")

if __name__ == "__main__":
    main()

