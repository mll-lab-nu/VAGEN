import torch
from PIL import Image
import os
from dreamsim import dreamsim
from typing import List, Any


class DreamSimScoreCalculator:
    """DreamSim perceptual similarity scorer."""

    def __init__(self, pretrained=True, cache_dir="~/.cache", device=None):
        cache_dir = os.path.expanduser(cache_dir)
        self.device = device or "cpu"
        self.model, self.preprocess = dreamsim(
            pretrained=pretrained, cache_dir=cache_dir, device=self.device
        )

    def calculate_similarity_score(self, gt_im, gen_im):
        """Similarity = 1 - distance.  Returns [0, 1]."""
        img1 = self.preprocess(gt_im).to(self.device)
        img2 = self.preprocess(gen_im).to(self.device)
        with torch.no_grad():
            distance = self.model(img1, img2).item()
        return 1.0 - min(1.0, max(0.0, distance))

    def calculate_batch_scores(self, gt_images: List[Any], gen_images: List[Any]) -> List[float]:
        """Per-pair similarity (DreamSim has no native batch comparison)."""
        if not gt_images or not gen_images:
            return []
        gt_processed = [self.preprocess(img).to(self.device) for img in gt_images]
        gen_processed = [self.preprocess(img).to(self.device) for img in gen_images]
        scores = []
        for i in range(len(gt_processed)):
            with torch.no_grad():
                distance = self.model(gt_processed[i], gen_processed[i]).item()
            scores.append(1.0 - min(1.0, max(0.0, distance)))
        return scores
