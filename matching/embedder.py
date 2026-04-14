# matching/embedder.py

from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    """
    Embedding handler using SentenceTransformer.
    Loads model once and reuses it across all calls.
    show_progress_bar=False on every encode call — no tqdm bars.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str):
        """
        Generate embedding for a single text input.
        Returns None for empty input.
        """
        if not text or not text.strip():
            return None

        return self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False      # ← suppress tqdm bar for single encode
        )

    def get_batch_embeddings(self, texts: list):
        """
        Generate embeddings for a list of texts in a single batch pass.
        Returns a 2D tensor of shape (len(texts), embedding_dim).
        """
        if not texts:
            return torch.zeros((0, 384))   # 384 = all-MiniLM-L6-v2 dim

        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False       # ← suppress tqdm bar for batch encode
        )