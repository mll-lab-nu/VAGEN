"""
Utility functions for logging, including wandb setup.
"""

import os
import time
from typing import Dict

def setup_output_dir(output_dir: str) -> str:
    """
    Create output directory with timestamp.

    Args:
        output_dir: Base output directory

    Returns:
        Path to created output directory
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    full_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(full_output_dir, exist_ok=True)
    return full_output_dir

def setup_wandb(args, model_config, inference_config) -> None:
    """
    Set up wandb logging if enabled.

    Args:
        args: Command line arguments
        model_config: Model configuration
        inference_config: Inference configuration
    """
    try:
        import wandb

        # Extract wandb configuration
        wandb_config = inference_config.get("wandb", {})
        project_name = wandb_config.get("project", "vagen-inference")

        # Construct a descriptive run name
        model_name = model_config.get("name", "unknown-model")
        split_name = args.split
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{model_name}_{split_name}_{timestamp}"

        # Initialize wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model_config": model_config,
                "inference_config": inference_config,
                "args": vars(args)
            }
        )

        print(f"Initialized wandb logging: project={project_name}, run={run_name}")
    except ImportError:
        print("wandb not installed. Skipping wandb initialization.")
    except Exception as e:
        print(f"Error initializing wandb: {e}")

def maybe_log_val_generations_to_wandb(log_rst, val_generations_to_log_to_wandb=5, global_steps=0):
    """
    Log a table of validation samples with multiple images per sample to wandb.
    
    Args:
        log_rst: List of result dictionaries
        val_generations_to_log_to_wandb: Number of examples to log
        global_steps: Current global step
    """
    if val_generations_to_log_to_wandb <= 0:
        return

    try:
        import wandb
        import numpy as np
        
        # Check if wandb is initialized
        if not wandb.run:
            print('WARNING: wandb run not found, but val_generations_to_log_to_wandb is set. Skipping logging.')
            return
            
        print(f"Logging {val_generations_to_log_to_wandb} validation examples to wandb")
        
        # Extract data from results
        inputs = []
        outputs = []
        scores = []
        images = []
        
        for item in log_rst:
            inputs.append(item['config_id'])
            outputs.append(item['output_str'])
            scores.append(item['metrics'].get('score', 0))
            # Check if image_data exists
            if 'image_data' in item:
                images.append(item['image_data'])
            else:
                images.append(None)
        
        # Handle the case where images might not be provided
        if all(img is None for img in images):
            samples = list(zip(inputs, outputs, scores))
            has_images = False
            max_images_per_sample = 0
        else:
            # Here, images is expected to be a list of lists, where each inner list contains images for one sample
            samples = list(zip(inputs, outputs, scores, images))
            has_images = True
            # Find maximum number of images in any sample
            max_images_per_sample = max(len(img_list) if isinstance(img_list, (list, tuple)) else 1 for img_list in images if img_list is not None)
        
        # Sort samples by input text
        samples.sort(key=lambda x: x[0])
        
        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        rng.shuffle(samples)
        
        # Take first N samples after shuffling
        samples = samples[:val_generations_to_log_to_wandb]
        
        # Create column names for all samples
        if has_images:
            columns = ["step"]
            for i in range(len(samples)):
                columns.extend([f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"])
                columns.extend([f"image_{i+1}_{j+1}" for j in range(max_images_per_sample)])
        else:
            columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])
        
        # Try to get existing table or create new one
        validation_table = None
        try:
            # Try to access the table if it exists in wandb
            validation_table = wandb.run.summary.get("val/generations_table")
        except:
            pass
        
        if validation_table is None:
            # Initialize the table on first call
            validation_table = wandb.Table(columns=columns)
            table_data = []
        else:
            # Use existing data
            table_data = validation_table.data
        
        # Create a new table with same columns and existing data
        new_table = wandb.Table(columns=columns, data=table_data)
        
        # Add new row with all data
        row_data = []
        row_data.append(global_steps)
        
        for sample in samples:
            if has_images:
                if len(sample) == 4:  # With image
                    input_text, output_text, score, sample_images = sample
                    row_data.extend([input_text, output_text, score])
                    
                    # Handle if sample_images is a single image or list of images
                    if sample_images is None:
                        # No images for this sample
                        row_data.extend([None] * max_images_per_sample)
                    else:
                        if not isinstance(sample_images, (list, tuple)):
                            sample_images = [sample_images]
                        
                        # Convert each image to wandb.Image
                        wandb_images = []
                        for img in sample_images:
                            try:
                                if img is not None and not isinstance(img, wandb.Image):
                                    img = wandb.Image(img)
                                wandb_images.append(img)
                            except Exception as e:
                                print(f"Error converting image to wandb.Image: {e}")
                                wandb_images.append(None)
                        
                        # Pad with None if there are fewer images than max_images_per_sample
                        wandb_images.extend([None] * (max_images_per_sample - len(wandb_images)))
                        row_data.extend(wandb_images)
                else:  # Without image
                    row_data.extend(sample)
                    row_data.extend([None] * max_images_per_sample)
            else:
                row_data.extend(sample)
        
        # Add the data to the table
        try:
            new_table.add_data(*row_data)
            # Log the table
            wandb.log({"val/generations": new_table}, step=global_steps)
            print(f"Successfully logged {len(samples)} examples to wandb table 'val/generations'")
        except Exception as e:
            print(f"Error adding data to table: {e}")
            
    except ImportError:
        print("wandb not installed. Skipping validation example logging.")
    except Exception as e:
        print(f"Error in maybe_log_val_generations_to_wandb: {e}")