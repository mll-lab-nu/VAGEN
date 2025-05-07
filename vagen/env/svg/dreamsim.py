import torch
from PIL import Image
import os
from dreamsim import dreamsim
import logging


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

    def calculate_batch_scores(self, gt_images, gen_images):
        """
        Calculate similarity scores for a batch of image pairs.
        """
        # Preprocess all images
        gt_processed = [self.preprocess(img) for img in gt_images]
        gen_processed = [self.preprocess(img) for img in gen_images]

        scores = []
        # Process each pair
        for gt, gen in zip(gt_processed, gen_processed):
            # Move to device
            gt = gt.to(self.device)
            gen = gen.to(self.device)

            # Calculate distance
            with torch.no_grad():
                distance = self.model(gt, gen).item()

            # Convert to similarity score
            similarity = 1.0 - min(1.0, max(0.0, distance))
            scores.append(similarity)

        return scores


# Compatibility function for existing code
def get_dreamsim_model(device="cuda:0"):
    """
    Create a new DreamSim model instance.
    This function exists for backward compatibility.
    The service should use DreamSimScoreCalculator directly.
    """
    logging.info(f"Creating new DreamSim model on {device}")
    return DreamSimScoreCalculator(device=device)