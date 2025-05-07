import torch
from PIL import Image
import os
from dreamsim import dreamsim
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

class DreamSimScoreCalculator:
    """
    A wrapper class for DreamSim model to calculate similarity scores between images.
    """

    def __init__(self, pretrained=True, cache_dir="~/.cache", device=None):
        """
        Initialize DreamSim model.
        """
        cache_dir = os.path.expanduser(cache_dir)

        # Verify device availability
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        # Load model and preprocessor
        self.model, self.preprocess = dreamsim(pretrained=pretrained, cache_dir=cache_dir, device=self.device)

    def calculate_similarity_score(self, gt_im, gen_im):
        """
        Calculate similarity score between ground truth and generated images.
        """
        # Preprocess images
        img1 = self.preprocess(gt_im)
        img2 = self.preprocess(gen_im)

        # Move to device if necessary
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # Calculate distance (lower is better)
        with torch.no_grad():
            distance = self.model(img1, img2).item()

        # Convert distance to similarity score (1 - normalized distance)
        similarity = 1.0 - min(1.0, max(0.0, distance))

        return similarity

    def calculate_batch_scores(self, gt_images: List[Any], gen_images: List[Any]) -> List[float]:
        """
        Calculate similarity scores for multiple image pairs.
        Since DreamSim doesn't natively support batch comparison, we process each pair individually.
        """
        if not gt_images or not gen_images:
            return []
            
        batch_size = len(gt_images)
        
        gt_processed = [self.preprocess(img).to(self.device) for img in gt_images]
        gen_processed = [self.preprocess(img).to(self.device) for img in gen_images]
        
        scores = []
        for i in range(batch_size):
            with torch.no_grad():
                distance = self.model(gt_processed[i], gen_processed[i]).item()
            similarity = 1.0 - min(1.0, max(0.0, distance))
            scores.append(similarity)
        
        return scores