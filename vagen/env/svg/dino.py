import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BaseMetric:
    def __init__(self):
        self.meter = AverageMeter()

    def reset(self):
        self.meter.reset()
        
    def calculate_score(self, batch, update=True):
        """
        Batch: {"gt_im": [PIL Image], "gen_im": [Image]}
        """
        values = []
        batch_size = len(next(iter(batch.values())))
        for index in range(batch_size):
            kwargs = {}
            for key in ["gt_im", "gen_im", "gt_svg", "gen_svg", "caption"]:
                if key in batch:
                    kwargs[key] = batch[key][index]
            try:
                measure = self.metric(**kwargs)
            except Exception as e:
                print(f"Error calculating metric: {e}")
                continue
            if math.isnan(measure):
                continue
            values.append(measure)

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan")

        score = sum(values) / len(values)
        if update:
            self.meter.update(score, len(values))
            return self.meter.avg, values
        else:
            return score, values

    def metric(self, **kwargs):
        """This method should be overridden by subclasses"""
        raise NotImplementedError("The metric method must be implemented by subclasses.")
    
    def get_average_score(self):
        return self.meter.avg


class DINOScoreCalculator(BaseMetric): 
    def __init__(self, config=None, model_size='large', device='cuda:0'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.model_size = model_size
        self.model, self.processor = self.get_DINOv2_model(model_size)
        self.device = device
        self.model = self.model.to(self.device)
        self.metric = self.calculate_DINOv2_similarity_score

    def get_DINOv2_model(self, model_size):
        if model_size == "small":
            model_size = "facebook/dinov2-small"
        elif model_size == "base":
            model_size = "facebook/dinov2-base"
        elif model_size == "large":
            model_size = "facebook/dinov2-large"
        else:
            raise ValueError(f"model_size should be either 'small', 'base' or 'large', got {model_size}")
        return AutoModel.from_pretrained(model_size), AutoImageProcessor.from_pretrained(model_size)

    def process_input(self, image, processor):
        """Process images efficiently in batches when possible"""
        if isinstance(image, list):
            if all(isinstance(img, Image.Image) for img in image):
                # Process all images in a single batch to maximize GPU utilization
                with torch.no_grad():
                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)
                return features
            else:
                features_list = []
                for img in image:
                    features_list.append(self.process_input(img, processor))
                return torch.cat(features_list, dim=0)
        
        if isinstance(image, str):
            image = Image.open(image)
            
        if isinstance(image, Image.Image):
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            return features
        elif isinstance(image, torch.Tensor):
            features = image.unsqueeze(0) if image.dim() == 1 else image
            return features
        else:
            raise ValueError("Input must be a file path, PIL Image, or tensor of features")

    def calculate_DINOv2_similarity_score(self, **kwargs):
        """Calculate similarity score between two images"""
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('gen_im')
        features1 = self.process_input(image1, self.processor)
        features2 = self.process_input(image2, self.processor)

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(features1, features2).item()
        sim = (sim + 1) / 2  # Convert from [-1, 1] to [0, 1] range

        return sim
    
    def calculate_batch_scores(self, gt_images: List[Any], gen_images: List[Any]) -> List[float]:
        """
        Calculate similarity scores for multiple image pairs in a single batch.
        DINO can process all images in a batch efficiently.
        """      
        if not gt_images: 
            return []
        
        gt_features = self.process_input(gt_images, self.processor)
        gen_features = self.process_input(gen_images, self.processor)
        
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(gt_features, gen_features)
        
        scores = [(sim.item() + 1) / 2 for sim in similarities]
        
        return scores