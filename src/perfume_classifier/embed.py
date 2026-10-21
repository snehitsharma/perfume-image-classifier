"""Image embeddings from a pretrained OpenCLIP model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class _Images(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB"))


class Embedder:
    def __init__(self, model_name: str, pretrained: str, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(self.device)
        self.name = f"{model_name}/{pretrained}"

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        return self._encode(batch)

    @torch.no_grad()
    def embed_paths(self, paths: list[Path], batch_size: int = 128, workers: int = 4) -> np.ndarray:
        loader = DataLoader(_Images(paths, self.transform), batch_size=batch_size, num_workers=workers)
        return np.vstack([self._encode(batch.to(self.device)) for batch in loader])

    def _encode(self, batch: torch.Tensor) -> np.ndarray:
        with torch.autocast(self.device, enabled=self.device == "cuda"):
            feats = self.model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)
