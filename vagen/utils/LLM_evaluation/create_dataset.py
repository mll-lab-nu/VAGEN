import pandas as pd
import numpy as np
import os
import random
from pathlib import Path

def create_test_dataset(env_name, sample_count):
    """
    Create test datasets by sampling steps from all raw CSV files and saving them in the
    environment-specific directory.
    
    Args:
        env_name (str): The environment name (e.g., 'navigation') used for output directory.
        sample_count (int): Number of samples to extract from each CSV file.
    """
    # Define paths
    raw_data_dir = Path(f"raw_data/{env_name}")
    output_dir = Path(f"data/{env_name}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files in the raw_data directory
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {raw_data_dir}")
    
    # Process each CSV file
    for csv_file in csv_files:
        # Get base name without extension
        base_name = csv_file.stem
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get the total number of steps
        total_steps = len(df)
        
        if sample_count > total_steps:
            print(f"Warning: Requested sample count ({sample_count}) for {base_name} exceeds available steps ({total_steps}). Using all steps.")
            file_sample_count = total_steps
        else:
            file_sample_count = sample_count
        
        # Evenly distribute the step indices to sample
        step_indices = np.linspace(0, total_steps - 1, file_sample_count, dtype=int).tolist()
        
        # Initialize a list to store sampled data
        sampled_data = []
        
        for step_idx in step_indices:
            step_row = df.iloc[step_idx]
            # Convert numpy.int64 to regular Python int to avoid serialization issues
            step_num = int(step_row['step'])
            
            # Find all valid samples for this step
            valid_samples = []
            
            # Loop through each sample column group
            for i in range(1, 9):  # Assuming samples 1-8
                sample_id_col = f"sample_{i}_id"
                
                # Check if this sample has an ID (indicating it's valid)
                if sample_id_col in step_row and pd.notna(step_row[sample_id_col]):
                    sample_data = {
                        "step": step_num,
                        "id": step_row[f"sample_{i}_id"],
                        "env_name": step_row[f"sample_{i}_env_name"],
                        "prompt": step_row[f"sample_{i}_prompt"],
                        "response": step_row[f"sample_{i}_response"],
                        "parsed_answer": step_row[f"sample_{i}_parsed_answer"]
                    }
                    valid_samples.append(sample_data)
            
            # If valid samples found for this step, randomly choose one
            if valid_samples:
                sampled_data.append(random.choice(valid_samples))
        
        # Create a DataFrame from the sampled data
        if sampled_data:
            sampled_df = pd.DataFrame(sampled_data)
            
            # Generate output filename
            output_file = output_dir / f"{base_name}.csv"
            
            # Save the data to CSV file
            sampled_df.to_csv(output_file, index=False)
            
            print(f"Created test dataset with {len(sampled_data)} samples at {output_file}")
        else:
            print(f"Warning: No valid samples found for {base_name}")

def main():
    """
    Main function to run the script with command line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test datasets by sampling from raw data')
    parser.add_argument('env_name', type=str, help='Environment name (e.g., navigation)')
    parser.add_argument('sample_count', type=int, help='Number of samples to extract from each file')
    
    args = parser.parse_args()
    
    create_test_dataset(args.env_name, args.sample_count)

if __name__ == "__main__":
    main()