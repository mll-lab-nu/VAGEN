import torch
import torch.nn as nn
from typing import List, Any
from transformers import AutoModel, AutoImageProcessor
from PIL import Image


class DINOScoreCalculator:
    """DINOv2-based image similarity scorer."""

    def __init__(self, model_size='large', device='cuda:0'):
        self.model_size = model_size
        self.device = device
        self.model, self.processor = self._load_model(model_size)
        self.model = self.model.to(self.device)

    @staticmethod
    def _load_model(model_size):
        model_name = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large",
        }.get(model_size)
        if model_name is None:
            raise ValueError(f"model_size must be 'small', 'base', or 'large', got {model_size}")
        return AutoModel.from_pretrained(model_name), AutoImageProcessor.from_pretrained(model_name)

    def _extract_features(self, images):
        """Extract features from a single image or list of images."""
        if not isinstance(images, list):
            images = [images]
        # Open file paths if needed
        images = [Image.open(img) if isinstance(img, str) else img for img in images]
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        return features

    def calculate_DINOv2_similarity_score(self, gt_im, gen_im):
        """Calculate cosine similarity between two images. Returns [0, 1]."""
        feat1 = self._extract_features(gt_im)
        feat2 = self._extract_features(gen_im)
        cos = nn.CosineSimilarity(dim=1)
        sim = cos(feat1, feat2).item()
        return (sim + 1) / 2

    def calculate_batch_scores(self, gt_images: List[Any], gen_images: List[Any]) -> List[float]:
        """Batch similarity scores for multiple image pairs."""
        if not gt_images:
            return []
        gt_features = self._extract_features(gt_images)
        gen_features = self._extract_features(gen_images)
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(gt_features, gen_features)
        return [(sim.item() + 1) / 2 for sim in similarities]
