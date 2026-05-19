import io
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from src.config import DINO_MODEL_ID

class DinoEmbedder:
    def __init__(self):
        print(f"Loading DINOv2 model: {DINO_MODEL_ID}")

        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        self.model = AutoModel.from_pretrained(DINO_MODEL_ID)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"  → Running on: {self.device}")

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray | None:
        """
        Convert raw image bytes (JPEG/PNG/WEBP) to a DINOv2 embedding vector.

        Returns a 1-D numpy float32 array of shape (embedding_dim,),
        or None if the image cannot be decoded.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"  [warn] Could not decode image: {e}")
            return None

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        return embedding.cpu().float().numpy()

    def embed_batch(self, images_bytes: list[bytes]) -> list[np.ndarray | None]:
        """
        Embed a list of raw image byte strings in one forward pass.
        More efficient than calling embed_bytes in a loop.
        """
        pil_images = []
        valid_indices = []

        for i, b in enumerate(images_bytes):
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                pil_images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"  [debug] Failed to decode image: {e}")
                print(f"  [debug] First 50 bytes: {b[:50]}")

        if not pil_images:
            return [None] * len(images_bytes)

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings_tensor = outputs.last_hidden_state[:, 0, :].cpu().float()

        result: list[np.ndarray | None] = [None] * len(images_bytes)
        for batch_idx, original_idx in enumerate(valid_indices):
            result[original_idx] = embeddings_tensor[batch_idx].numpy()

        return result

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension for the loaded model."""
        dims = {
            "facebook/dinov2-small": 384,
            "facebook/dinov2-base": 768,
            "facebook/dinov2-large": 1024,
            "facebook/dinov2-giant": 1536,
        }
        return dims.get(DINO_MODEL_ID, 768)
