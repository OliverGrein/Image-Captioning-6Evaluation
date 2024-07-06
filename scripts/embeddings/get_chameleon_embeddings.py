from chameleon.inference.chameleon import ChameleonInferenceModel, TokenManager, DistributedMode, Options, Transformer
from chameleon.inference import loader
import torch
import PIL


class CustomTokenManager(TokenManager):
    def __init__(self, tokenizer_path: str, vqgan_cfg_path: str, vqgan_ckpt_path: str, device: str | None = None):
        super().__init__(tokenizer_path, vqgan_cfg_path, vqgan_ckpt_path, device=device)
        

    def tokenize_text_and_image(self, text: str, img: PIL.Image) -> list[int]:
        tokens = self.tokenize_text(text)
        tokens += self.tokenize_image(img)
        return tokens

class CustomChameleonInferenceModel(ChameleonInferenceModel):
    def __init__(
        self,
        model: Transformer | str,
        tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str,
        options: Options | None = None,
        distributed_mode: DistributedMode = DistributedMode.AUTO,
    ):
        super().__init__(model, tokenizer_path, vqgan_cfg_path, vqgan_ckpt_path, options=options, distributed_mode=distributed_mode)
        self.model = loader.load_model(model)
        self.token_manager = CustomTokenManager(tokenizer_path, vqgan_cfg_path, vqgan_ckpt_path, device="cuda")

    def combined_embeddings(self, text: str, img: PIL.Image) -> torch.Tensor:
        tokens = self.token_manager.tokenize_text_and_image(text, img)
        input_tensor = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            hidden_states = outputs.hidden_states  # Extract hidden states from the model output

        embeddings = hidden_states[-1]  # Use the last hidden state as the embeddings

        return embeddings

def main():
    model = CustomChameleonInferenceModel(
        "meta-chameleon-7b/models/7b/",
        "meta-chameleon-7b/tokenizer/text_tokenizer.json",
        "meta-chameleon-7b/tokenizer/vqgan.yaml",
        "meta-chameleon-7b/tokenizer/vqgan.ckpt",
    )

    # Load an image and a caption
    image_path = 'Image-Captioning-6Evaluation/data/flickr-8k/Images/667626_18933d713e.jpg'
    caption = 'A beautiful sunrise over the mountains.'
    image = PIL.Image.open(image_path)

    # Get the combined embeddings
    combined_embeddings = model.combined_embeddings(text=caption, img=image)
    print("Combined Embeddings Shape:", combined_embeddings.shape)

if __name__ == "__main__":
    main()