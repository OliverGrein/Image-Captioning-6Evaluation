import pandas as pd
from tqdm import tqdm
import cohere
from dotenv import load_dotenv
import os




def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    model="embed-english-v3.0"
    desc="Embedding chunks"
    chunk_size = 128

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv('COHERE_API_KEY')
    
    flickr8k_captions = pd.read_csv("Image-Captioning-6Evaluation/data/flickr-8k/processed/flickr8k_captions.csv")
    mscoco_captions = pd.read_csv("Image-Captioning-6Evaluation/data/mscoco/processed/captions.csv")
    pascal50s_captions = pd.read_csv("Image-Captioning-6Evaluation/data/pascal50s/raw/captions.csv")

    # Split captions into chunks that the APi can handle
    caption_chunks_flickr8k = list(split_into_chunks(flickr8k_captions['caption'].tolist(), chunk_size))
    caption_chunks_mscoco = list(split_into_chunks(mscoco_captions['caption'].tolist(), chunk_size))
    caption_chunks_pascal50s = list(split_into_chunks(pascal50s_captions['caption'].tolist(), chunk_size))
    co = cohere.Client(api_key)
    # Get embeddings for each chunk
    flickr8k_embeddings = []
    #for chunk in tqdm(caption_chunks_flickr8k, desc=desc):
    #    embeddings = co.embed(texts=chunk, model=model,input_type="search_document").embeddings
    #    flickr8k_embeddings.extend(embeddings)

    mscoco_embeddings = []
    for chunk in tqdm(caption_chunks_mscoco, desc=desc):
        embeddings = co.embed(texts=chunk, model=model,input_type="search_document").embeddings
        mscoco_embeddings.extend(embeddings)

    pascal50s_embeddings = []
    for chunk in tqdm(caption_chunks_pascal50s, desc=desc):
        embeddings = co.embed(texts=chunk, model=model,input_type="search_document").embeddings
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