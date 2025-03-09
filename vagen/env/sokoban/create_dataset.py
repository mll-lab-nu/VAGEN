"""
Preprocess dataset for genereal tasks
"""

import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets

from vagen.env.create_dataset import DatasetCreator

class SokobanDatasetCreator(DatasetCreator):
    
    def create_filtered_dataset(
        self,
        start_seed: int,
        train_size: int,
        test_size: int,
        max_steps: int = 5
    ):
        # TODO
        return 






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='vagen/env/sokoban.yaml')
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=100)
    args = parser.parse_args()
    creator = SokobanDatasetCreator(config_path=args.config_path)
    creator.create_filtered_dataset(start_seed=args.start_seed, train_size=args.train_size, test_size=args.test_size)
