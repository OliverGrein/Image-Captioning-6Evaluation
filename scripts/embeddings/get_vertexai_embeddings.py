import pandas as pd
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from dotenv import load_dotenv
import vertexai
import google.auth
import os


def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_embeddings(captions, output_file, model, task, chunk_size, desc):
    ensure_dir(output_file)
    caption_chunks = list(split_into_chunks(captions["caption"].tolist(), chunk_size))

    embeddings = []
    for chunk in tqdm(caption_chunks, desc=desc):
        inputs = [TextEmbeddingInput(text, task) for text in chunk]
        chunk_embeddings = model.get_embeddings(inputs)
        embeddings.extend([embedding.values for embedding in chunk_embeddings])

    # Add embeddings as a new column
    captions["embedding"] = embeddings

    # Save the updated DataFrame to CSV
    captions.to_csv(output_file, index=False)


def main():
    model = "textembedding-gecko@003"
    task = "RETRIEVAL_DOCUMENT"
    chunk_size = 128

    # Load environment variables from .env file
    load_dotenv()

    # Set up Google Cloud authentication
    credentials, project_id = google.auth.default()
    vertexai.init(project=project_id, credentials=credentials)

    flickr8k_captions = pd.read_csv("data/flickr-8k/processed/flickr8k_captions.csv")
    mscoco_captions = pd.read_csv("data/mscoco/processed/captions.csv")
    pascal50s_captions = pd.read_csv("data/pascal50s/raw/captions.csv")

    model = TextEmbeddingModel.from_pretrained(model)

    process_embeddings(
        flickr8k_captions,
        "data/vertexai_embeddings/flickr8k_embeddings.csv",
        model,
        task,
        chunk_size,
        "Embedding Flickr8k chunks",
    )
    process_embeddings(
        mscoco_captions,
        "data/vertexai_embeddings/mscoco_embeddings.csv",
        model,
        task,
        chunk_size,
        "Embedding MSCOCO chunks",
    )
    process_embeddings(
        pascal50s_captions,
        "data/vertexai_embeddings/pascal50s_embeddings.csv",
        model,
        task,
        chunk_size,
        "Embedding Pascal50S chunks",
    )

    print("Done")


if __name__ == "__main__":
    main()
