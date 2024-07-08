import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from tokenizers import Tokenizer
from chameleon.inference.chameleon import ImageTokenizer
import numpy as np

def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    # Load image tokenizer
    image_tokenizer = ImageTokenizer("meta-chameleon-7b/tokenizer/vqgan.yaml", "meta-chameleon-7b/tokenizer/vqgan.ckpt")

    # Load captions from datasets and set path to dataset images
    flickr8k_captions = pd.read_csv("Image-Captioning-6Evaluation/data/flickr-8k/processed/flickr8k_captions.csv")
    mscoco_captions = pd.read_csv("Image-Captioning-6Evaluation/data/mscoco/processed/captions.csv")
    pascal50s_captions = pd.read_csv("Image-Captioning-6Evaluation/data/pascal50s/raw/captions.csv")
    pascal_image_path = "Image-Captioning-6Evaluation/data/pascal50s/raw/images/"
    flickr8k_image_path = "Image-Captioning-6Evaluation/data/flickr-8k/Images/"
    mscoco_image_path = "Image-Captioning-6Evaluation/data/mscoco/Images/"


    # Use embedding chunks as the image embeddings get really large
    chunk_size = 500
    desc = "Embedding chunks"
    caption_chunks_flickr8k = list(split_into_chunks(flickr8k_captions['caption'].tolist(), chunk_size))
    caption_chunks_mscoco = list(split_into_chunks(mscoco_captions['caption'].tolist(), chunk_size))
    caption_chunks_pascal50s = list(split_into_chunks(pascal50s_captions['caption'].tolist(), chunk_size))
    
    def process_chunk(chunk, image_paths, chunk_id=0):
        """
        processes a chunk of images and returns the corresponding embeddings.

        Args:
            chunk (list of str): A list of text strings to determine the chunk size. It is not directly used in the function other than for calculating the chunk length.
            image_paths (list of str): A list of paths to images to be processed.
            chunk_id (int, optional): The index of the current chunk being processed. Defaults to 0.
        
        Returns:
            torch.Tensor: A tensor containing the embeddings from the processed images.
        """
        c_len = len(chunk)
        max_img_id = min(len(image_paths),c_len* (chunk_id+1) )
        image_paths = image_paths[c_len * chunk_id:max_img_id]
        stackable_tensor = [
            image_tokenizer.img_tokens_from_pil(Image.open(path))
            for path in image_paths
        ]
        image_tokens = torch.stack(stackable_tensor)

        return image_tokenizer.img_embeddings_from_pil(image_tokens) 
     
    # Create image embeddings for flickr-8k dataset
    for chunk_id,chunk in enumerate(tqdm(caption_chunks_flickr8k, desc=desc)):
        image_paths = [os.path.join(flickr8k_image_path, img) for img in flickr8k_captions['image'].str[:-2]]
        embeddings = process_chunk(chunk, image_paths,chunk_id)
        new_embeddings = np.array(embeddings)
        np.save(f"Image-Captioning-6Evaluation/data/flickr-8k/image_emb/flickr-8k_embeddings_{chunk_id}.npy", new_embeddings)

    # Create image embeddings for pascal50s dataset
    for chunk_id,chunk in enumerate(tqdm(caption_chunks_pascal50s, desc=desc)):
        image_paths = [os.path.join(pascal_image_path, img) for img in pascal50s_captions['image']]
        embeddings = process_chunk(chunk, image_paths,chunk_id)
        new_embeddings = np.array(embeddings)
        np.save(f"Image-Captioning-6Evaluation/data/pascal50s/image_emb/pascal50s_embeddings_{chunk_id}.npy", new_embeddings)

    # Create image embeddings for mscoco dataset
    for chunk_id,chunk in enumerate(tqdm(caption_chunks_mscoco, desc=desc)):
        image_paths = [os.path.join(mscoco_image_path, img) for img in mscoco_captions['image_id'].astype(str).str.zfill(12) + ".jpg"]
        embeddings = process_chunk(chunk, image_paths,chunk_id)
        new_embeddings = np.array(embeddings)
        np.save(f"Image-Captioning-6Evaluation/data/mscoco/image_emb/mscoco_embeddings_{chunk_id}.npy", new_embeddings)

if __name__ == "__main__":
    main()