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
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--force-gen', action='store_true', 
                        help='Force dataset generation even if files already exist')
    # Added arguments based on the YAML config
    parser.add_argument('--dim_room', type=int, nargs=2, default=[6, 6],
                        help='Dimensions of the room [height, width]')
    parser.add_argument('--num_boxes', type=int, default=1,
                        help='Number of boxes in the environment')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of steps allowed')
    parser.add_argument('--search_depth', type=int, default=30,
                        help='Search depth that affects the starting position of the player')
    parser.add_argument('--visual_env', action='store_true',
                        help='Whether to use visual environment')
    parser.add_argument('--data_dir', type=str, default='data/sokoban',)

    args = parser.parse_args()
    args.name = 'sokoban'
    args.env_config = {
        'dim_room': args.dim_room,
        'num_boxes': args.num_boxes,
        'max_steps': args.max_steps,
        'search_depth': args.search_depth,
        'visual_env': args.visual_env
    }
    creator = SokobanDatasetCreator(config=vars(args))
    #creator.create_filtered_dataset(start_seed=args.start_seed, train_size=args.train_size, test_size=args.test_size)
    creator.create_dataset(start_seed=args.start_seed, train_size=args.train_size, test_size=args.test_size)
