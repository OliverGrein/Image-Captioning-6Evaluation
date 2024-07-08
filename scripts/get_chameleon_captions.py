# This script was part of a different idea that we did not follow through with, but we included in the github for completeness

from chameleon.inference.chameleon import ChameleonInferenceModel
import os
import pandas as pd

def main():
    # load chameleon model
    model = ChameleonInferenceModel(
        "meta-chameleon-7b/models/7b/",
        "meta-chameleon-7b/tokenizer/text_tokenizer.json",
        "meta-chameleon-7b/tokenizer/vqgan.yaml",
        "meta-chameleon-7b/tokenizer/vqgan.ckpt",
    )

    # Create captions for pascal 50s dataset
    pascal_data = []
    for filename in os.listdir("Image-Captioning-6Evaluation/data/pascal50s/raw/images/"):
        tokens = model.generate(
            prompt_ui=[
                {"type": "image", "value": f"file:Image-Captioning-6Evaluation/data/pascal50s/raw/images/{filename}"},
                {"type": "text", "value": "Create an image caption for this image, that describes the image. Do not add any opening sentences and do not make any judgements. Use a neutral tone."},
                {"type": "sentinel", "value": "<END-OF-TURN>"},
            ]
        )
        pascal_data.append([filename, model.decode_text(tokens)[0]])
        print(model.decode_text(tokens)[0])
    pascal_captions = pd.DataFrame(pascal_data, columns=["Image", "caption"])
    pascal_captions.to_csv("Image-Captioning-6Evaluation/data/pascal50s/chameleon_captions.csv")


    # Create captions for flickr-8k dataset
    flickr8k_data = []
    for filename in os.listdir("Image-Captioning-6Evaluation/data/flickr-8k/Images/"):
        tokens = model.generate(
            prompt_ui=[
                {"type": "image", "value": f"file:Image-Captioning-6Evaluation/data/flickr-8k/Images/{filename}"},
                {"type": "text", "value": "Create an image caption for this image."},
                {"type": "sentinel", "value": "<END-OF-TURN>"},
            ]
        )
        flickr8k_data.append([filename, model.decode_text(tokens)[0]])
        print(model.decode_text(tokens)[0])
    flickr_captions = pd.DataFrame(flickr8k_data, columns=["Image", "caption"])
    flickr_captions.to_csv("Image-Captioning-6Evaluation/data/flickr-8k/chameleon_captions.csv")

    # Create captions for mscoco dataset
    mscoco_data = []
    for filename in os.listdir("Image-Captioning-6Evaluation/data/mscoco/Images/"):
        tokens = model.generate(
            prompt_ui=[
                {"type": "image", "value": f"file:Image-Captioning-6Evaluation/data/mscoco/Images/{filename}"},
                {"type": "text", "value": "Create an image caption for this image."},
                {"type": "sentinel", "value": "<END-OF-TURN>"},
            ]
        )
        mscoco_data.append([filename, model.decode_text(tokens)[0]])
        print(model.decode_text(tokens)[0])
    mscoco_captions = pd.DataFrame(mscoco_data, columns=["Image", "caption"])
    mscoco_captions.to_csv("Image-Captioning-6Evaluation/data/mscoco/chameleon_captions.csv")


if __name__ == "__main__":
    main()