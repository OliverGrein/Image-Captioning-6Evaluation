import pandas as pd
from tqdm import tqdm
import voyageai
from dotenv import load_dotenv
import os


def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_embeddings(captions, output_file, model, chunk_size, desc):
    ensure_dir(output_file)
    caption_chunks = list(split_into_chunks(captions["caption"].tolist(), chunk_size))

    vo = voyageai.Client()
    embeddings = []
    for chunk in tqdm(caption_chunks, desc=desc):
        chunk_embeddings = vo.embed(chunk, model=model).embeddings
        embeddings.extend(chunk_embeddings)

    captions["embedding"] = embeddings
    captions.to_csv(output_file, index=False)


def main():
    model = "voyage-large-2-instruct"
    chunk_size = 128

    # Load API key from .env file
    load_dotenv()

    flickr8k_captions = pd.read_csv("data/flickr-8k/processed/flickr8k_captions.csv")
    mscoco_captions = pd.read_csv("data/mscoco/processed/captions.csv")
    pascal50s_captions = pd.read_csv("data/pascal50s/raw/captions.csv")

    process_embeddings(
        flickr8k_captions,
        "data/voyageai_embeddings/flickr8k_embeddings.csv",
        model,
        chunk_size,
        "Embedding Flickr8k chunks",
    )
    process_embeddings(
        mscoco_captions,
        "data/voyageai_embeddings/mscoco_embeddings.csv",
        model,
        chunk_size,
        "Embedding MSCOCO chunks",
    )
    process_embeddings(
        pascal50s_captions,
        "data/voyageai_embeddings/pascal50s_embeddings.csv",
        model,
        chunk_size,
        "Embedding Pascal50S chunks",
    )

    print("Done")


if __name__ == "__main__":
    main()
