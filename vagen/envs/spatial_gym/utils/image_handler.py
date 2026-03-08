"""
Simple image handler for spatial environments
"""
import os
import json
import re
import numpy as np
from typing import Dict, Union
from PIL import Image


class ImageHandler:
    """Handle images and data initialization based on position and orientation."""
    
    def __init__(self, base_dir: str = None, seed: int = None, image_dir: str = None, image_size: tuple = (512, 512), preload_images: bool = True):
        """
        Initialize image handler with data loading.
        
        Args:
            base_dir: Base directory containing data subdirectories (used with seed)
            seed: Random seed for directory selection (used with base_dir)
            image_dir: Direct path to image directory (e.g., '/path/to/data-test/run00')
                      If provided, base_dir and seed are ignored
            image_size: Target size for loaded images
            preload_images: Whether to load all images into memory
        """
        self.image_size = image_size
        self.preload_images = preload_images
        
        if image_dir is not None:
            # Direct initialization with image_dir
            self.image_dir = image_dir
            self.base_dir = os.path.dirname(image_dir)
            assert os.path.exists(self.image_dir), f"Image directory {self.image_dir} does not exist"
            with open(os.path.join(self.image_dir, "meta_data.json"), 'r') as f:
                self.json_data = json.load(f)
        else:
            # Traditional initialization with base_dir and seed
            assert base_dir is not None, "Either base_dir or image_dir must be provided"
            self.base_dir = base_dir
            self.image_dir, self.json_data = ImageHandler.load_data(base_dir, seed)
        
        self._update_mappings()
        self._image_map, self._image_path_map = self._load_images()
    
    def _update_mappings(self):
        """Update internal mappings from json_data."""
        self.objects = {obj['object_id']: obj for obj in self.json_data.get('objects', [])}
        tmp = {obj['name']: obj['object_id'] for obj in self.objects.values()}
        tmp.update({k.replace('_', ' '): v for k, v in tmp.items()})
        tmp['agent'] = 'agent'
        self.name_2_cam_id = tmp
    
    @staticmethod
    def load_data(base_dir: str, seed: int) -> tuple:
        """Load JSON data from a 'runNN' subdirectory (sorted by NN)."""
        target_run = f"run{seed:02d}"
        image_dir = os.path.join(base_dir, target_run)
        assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist"
        with open(os.path.join(image_dir, "meta_data.json"), 'r') as f:
            json_data = json.load(f)
            
        return image_dir, json_data

    def _load_images(self) -> Dict[str, Union[Image.Image, str]]:
        """Load images or paths based on preload setting."""
        image_map = {}
        image_path_map = {}
        
        # Construct mapping directly from available files in image_dir.
        # This supports both legacy 4-dir and new 8-dir datasets.
        cam_ids = list(self.objects.keys()) + ['agent']
        
        for cam_id in cam_ids:
            prefix = f"{cam_id}_facing_"
            for fname in os.listdir(self.image_dir):
                if not (fname.startswith(prefix) and fname.endswith(".png")):
                    continue
                key = fname[:-4]
                path = os.path.join(self.image_dir, fname)
                image_path_map[key] = path
                if self.preload_images:
                    image_map[key] = Image.open(path).resize(self.image_size, Image.LANCZOS)
        
        instruction_path = os.path.join(self.base_dir, 'instruction.png')
        assert os.path.exists(instruction_path)
        image_path_map['instruction'] = instruction_path
        if self.preload_images:
            image_map['instruction'] = Image.open(instruction_path).resize(self.image_size, Image.LANCZOS)            
        
        label_path = os.path.join(self.image_dir, 'orientation_instruction.png')
        assert os.path.exists(label_path)
        image_path_map['label'] = label_path
        if self.preload_images:
            image_map['label'] = Image.open(label_path).resize(self.image_size, Image.LANCZOS)

        return image_map, image_path_map
    
    def _normalize_direction(self, direction: Union[str, tuple]) -> str:
        """
        Normalize direction from tuple or string to one of 8 heading strings.
        
        Args:
            direction: Either a direction string or a tuple (dx, dz) vector.
                      or a tuple (dx, dz) representing direction vector
                      
        Returns:
            Direction string in {north, northeast, east, southeast, south, southwest, west, northwest}
        """
        def _from_tuple(dx: float, dz: float) -> str:
            vec = np.array([float(dx), float(dz)], dtype=float)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                raise ValueError(f"Invalid direction tuple: {(dx, dz)}")
            x, y = vec[0] / norm, vec[1] / norm
            angle = float(np.rad2deg(np.arctan2(x, y)))  # north=0, clockwise positive
            labels = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
            idx = int(round(angle / 45.0)) % 8
            return labels[idx]

        if isinstance(direction, str):
            d = direction.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
            aliases = {
                'n': 'north', 'north': 'north',
                'ne': 'northeast', 'northeast': 'northeast',
                'e': 'east', 'east': 'east',
                'se': 'southeast', 'southeast': 'southeast',
                's': 'south', 'south': 'south',
                'sw': 'southwest', 'southwest': 'southwest',
                'w': 'west', 'west': 'west',
                'nw': 'northwest', 'northwest': 'northwest',
            }
            if d not in aliases:
                raise ValueError(f"Invalid direction string: {direction}")
            return aliases[d]
        
        # Handle tuple (dx, dz) format
        dx, dz = direction
        return _from_tuple(dx, dz)
    
    def get_image(self, name: str = 'agent', direction: str = 'north') -> Image.Image:
        """
        Get image for given camera ID and direction.
        
        Args:
            name: Name of the object ('agent' or object_name or 'instruction' as string)
            direction: Direction string (supports 8 headings)
            
        Returns:
            PIL Image
            
        Raises:
            KeyError: If image not found
        """
        # Handle special static images that don't need direction
        if name in ['instruction', 'label']:
            key = name
        else:
            key = f"{self.name_2_cam_id[name]}_facing_{self._normalize_direction(direction)}"
        
        if key not in self._image_map:
            raise KeyError(f"Image not found for name '{name}' facing '{direction}'")
        
        if self.preload_images:
            return self._image_map[key]
        else:
            path = self._image_path_map[key]
            return Image.open(path).resize(self.image_size, Image.LANCZOS)
        
    def get_image_path(self, name: str = 'agent', direction: Union[str, tuple] = 'north') -> str:
        """
        Get image path for given camera ID and direction.
        
        Args:
            name: Name of the object ('agent' or object_name or 'instruction' as string)
                  Can also be position string like '2_4' for x=2, z=4
            direction: Direction string or tuple (dx, dz)
            
        Returns:
            Image file path
            
        Raises:
            KeyError: If image path not found
        """
        # Handle special static images that don't need direction
        if name in ['instruction', 'label']:
            key = name
        else:
            direction_str = self._normalize_direction(direction)
            # Try to resolve name to cam_id
            cam_id = self.name_2_cam_id[name] if name in self.name_2_cam_id else str(name[0]) + '_' + str(name[1])
            key = f"{cam_id}_facing_{direction_str}"
        
        if key not in self._image_path_map:
            # Try to construct path directly for position-based queries
            pos_path = os.path.join(self.image_dir, f"{key}.png")
            if os.path.exists(pos_path):
                return pos_path
            raise KeyError(f"Image path not found for name '{name}' facing '{direction}'")
        
        return self._image_path_map[key]
    