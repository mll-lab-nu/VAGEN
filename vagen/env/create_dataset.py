import numpy as np
import yaml
from datasets import Dataset, load_dataset
import os
import pandas as pd
import argparse
from pathlib import Path

class DatasetCreator:

    def __init__(self, config):
        self.config = config
        self.env_name = self.config['name']
        self.env_config = self.config['env_config']
        self.data_dir = self.config['data_dir']
        
        

    def create_dataset(self, start_seed, train_size, test_size, force_gen=False):
        train_file_path = os.path.join(self.data_dir, 'train.parquet')
        test_file_path = os.path.join(self.data_dir, 'test.parquet')
        
        # Check if files already exist and force_gen is False
        if not force_gen and os.path.exists(train_file_path) and os.path.exists(test_file_path):
            print(f"Dataset files already exist at {self.data_dir}. Skipping generation.")
            print(f"Use --force-gen to override and regenerate the dataset.")
            return
            
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        seeds = range(start_seed, start_seed + train_size + test_size)
        instructions = []
        for seed in seeds:
            # Generate instruction based on environment
            # This is a placeholder - actual implementation would depend on the environment
            instruction = f"Instruction for seed {seed}"
            instructions.append(instruction)
            
        def _create_instance(seed_idx, instruction):
            split = "train" if seed_idx < start_seed + train_size else "test"
            env_settings = {
                'env_name': self.env_name,
                'env_config': self.env_config,
                'seed': seed_idx
            }

            # TODO: no reward model defined here for the reward will be generated while rollout
            return {
                "data_source": self.env_name,
                "prompt": [{"role": "user", "content": instruction}],
                "extra_info": {"split": split, **env_settings}
            }

        train_instances = [_create_instance(start_seed + i, '') for i in range(train_size)]
        test_instances = [_create_instance(start_seed + train_size + i, '') for i in range(test_size)]
        
        train_dataset = Dataset.from_list(train_instances)
        test_dataset = Dataset.from_list(test_instances)

        def make_map_fn(split):
            def process_fn(example, idx):
                return example
            return process_fn

        
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        train_dataset.to_parquet(train_file_path)
        test_dataset.to_parquet(test_file_path)
        print(f"Dataset successfully generated at {self.data_dir}")


    def merge_parquet_files(
        self,
        source_files: list[str],
        output_file: str,
        columns: list[str] = None
    ):
        """
        Merge multiple parquet files into a single parquet file.
        
        Args:
            source_files (list): List of paths to parquet files to merge
            output_file (str): Path to save the merged parquet file
            columns (list, optional): List of columns to include in the merged file.
                                    If None, all columns are included.
        
        Returns:
            bool: True if successful, False otherwise
        """
        
        
        try:
            # Make sure the output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize an empty DataFrame to hold the merged data
            merged_df = pd.DataFrame()
            
            for file_path in source_files:
                # Check if the file exists
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} does not exist, skipping.")
                    continue
                    
                # Read the parquet file
                df = pd.read_parquet(file_path, columns=columns)
                
                # Append to the merged DataFrame
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            
            if merged_df.empty:
                print("No data to merge. Check if source files exist and contain data.")
                return False
                
            # Write the merged DataFrame to a parquet file
            merged_df.to_parquet(output_file, index=False)
            print(f"Successfully merged {len(source_files)} files into {output_file}")
            
            return True
            
        except Exception as e:
            print(f"Error merging parquet files: {str(e)}")
            return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    parser.add_argument('--start_seed', type=int, default=10000)
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--force-gen', action='store_true', 
                        help='Force dataset generation even if files already exist')
    args = parser.parse_args()
    creator = DatasetCreator(config_path=args.config_path)
    creator.create_dataset(
        start_seed=args.start_seed, 
        train_size=args.train_size, 
        test_size=args.test_size,
        force_gen=args.force_gen
    )