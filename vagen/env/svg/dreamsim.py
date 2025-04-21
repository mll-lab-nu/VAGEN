import torch
from PIL import Image
import os
from dreamsim import dreamsim

# Create global cache and lock, similar to DINO implementation
_model_cache = {}
_model_cache_lock = threading.Lock()
_model_counter = 0

def get_dreamsim_model(device=None):
    """
    Get a singleton instance of DreamSim model, using cache to avoid duplicate loading

    Args:
        device: Device to run model on

    Returns:
        DreamSimScoreCalculator: Instance of DreamSim calculator
    """
    global _model_counter

    # Choose device based on availability if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use device as cache key
    cache_key = f"dreamsim_{device}"

    with _model_cache_lock:
        if cache_key not in _model_cache:
            _model_counter += 1
            pid = os.getpid()
            logging.info(f"Process {pid}: Created DreamSim model #{_model_counter} on {device}")
            _model_cache[cache_key] = DreamSimScoreCalculator(device=device)
        return _model_cache[cache_key]

class DreamSimScoreCalculator:
    """
    A wrapper class for DreamSim model to calculate similarity scores between images.
    """
    def __init__(self, pretrained=True, cache_dir="~/.cache", device=None):
        """
        Initialize DreamSim model.
        
        Args:
            pretrained: Whether to use pretrained model
            cache_dir: Cache directory for model weights
            device: Device to run the model on (defaults to CUDA if available, else CPU)
        """
        cache_dir = os.path.expanduser(cache_dir)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model and preprocessor
        self.model, self.preprocess = dreamsim(pretrained=pretrained, cache_dir=cache_dir)
        self.model = self.model.to(self.device)
        
    def calculate_similarity_score(self, gt_im, gen_im):
        """
        Calculate similarity score between ground truth and generated images.
        
        Args:
            gt_im: Ground truth PIL Image
            gen_im: Generated PIL Image
            
        Returns:
            float: Similarity score (1 - distance, normalized to [0, 1])
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
        # DreamSim usually outputs values in range [0, 1] where lower means more similar
        # We invert it so that higher means more similar (1 = identical)
        similarity = 1.0 - min(1.0, max(0.0, distance))
        
        return similarity
    
    def calculate_batch_scores(self, gt_images, gen_images):
        """
        Calculate similarity scores for a batch of image pairs.
        
        Args:
            gt_images: List of ground truth PIL Images
            gen_images: List of generated PIL Images
            
        Returns:
            List[float]: List of similarity scores
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

# Helper function to get or initialize DreamSim model
def get_dreamsim_model(device=None):
    """
    Get an instance of DreamSim model.
    
    Args:
        device: Device to run model on
        
    Returns:
        DreamSimScoreCalculator: Instance of DreamSim calculator
    """
    return DreamSimScoreCalculator(device=device)