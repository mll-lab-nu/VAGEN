import json
import argparse
from pathlib import Path
import pandas as pd
import tiktoken

def get_token_count(text, model="gpt-4"):
    """
    Calculate the number of tokens in the text for a given model.
    
    Args:
        text (str): The text to count tokens for.
        model (str): The model name to use for token counting.
    
    Returns:
        int: Number of tokens in the text.
    """
    if not text:
        return 0
    
    try:
        # Get the encoding for the model
        # For most GPT models, we can use the cl100k_base encoding
        if "gpt-4" in model.lower() or "gpt-3.5" in model.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Fallback to a common encoding
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # Count tokens
        tokens = encoding.encode(text)
        return len(tokens)
    
    except Exception as e:
        print(f"Warning: Could not count tokens for model {model}: {e}")
        # Fallback to rough estimation (1 token ≈ 4 characters for English)
        return len(text) // 4

def read_jsonl(file_path):
    """
    Read data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file.
    
    Returns:
        list: List of dictionaries from the JSONL file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line.strip()))
    return data

def calculate_accuracy(data, groupby=None):
    """
    Calculate accuracy stats from the input data.
    
    Args:
        data (list): List of data records with human_answer and parsed_answer fields.
        groupby (str): Field to group by ('env', 'type', or None for both).
    
    Returns:
        DataFrame: Accuracy statistics.
    """
    # Prepare analysis data
    analysis_data = []
    
    for item in data:
        # Get model name (assume it's in the data or set default)
        model = item.get("model", "unknown")
        
        # Get other fields
        env = item.get("env", "")
        type_name = item.get("type", "")
        human_answer = item.get("human_answer", "")
        parsed_answer = item.get("parsed_answer", "")
        response = item.get("response", "")
        
        # Determine if the answer is correct
        # If parsed_answer is null/empty, it's considered wrong
        is_correct = False
        if parsed_answer and human_answer:
            is_correct = parsed_answer.upper() == human_answer.upper()
        
        # Calculate response token count using the model name
        response_token_count = get_token_count(response, model)
        
        # Add to analysis data
        analysis_data.append({
            "model": model,
            "env": env,
            "type": type_name,
            "human_answer": human_answer,
            "parsed_answer": parsed_answer,
            "is_correct": is_correct,
            "response_token_count": response_token_count
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(analysis_data)
    
    # Group by specified field(s)
    if groupby == 'env':
        groups = df.groupby(["model", "env"])
    elif groupby == 'type':
        groups = df.groupby(["model", "type"])
    else:  # Both env and type
        groups = df.groupby(["model", "env", "type"])
    
    # Calculate statistics - only the 4 required fields
    stats = groups.agg(
        correct=("is_correct", "sum"),
        response_len=("response_token_count", "mean"),
    )
    
    # Add total count and calculate accuracy
    stats["total"] = groups.size()
    stats["accuracy"] = (stats["correct"] / stats["total"]) * 100
    
    # Round the response length for readability
    stats["response_len"] = stats["response_len"].round(2)
    
    # Reorder columns to match the requested order: total, correct, response_len, accuracy
    stats = stats[["total", "correct", "response_len", "accuracy"]]
    
    return stats

def save_stats_to_jsonl(stats, output_file):
    """
    Save statistics to JSONL format.
    
    Args:
        stats (DataFrame): Statistics DataFrame.
        output_file (Path): Path to save the statistics.
    
    Returns:
        None
    """
    # Reset the index for saving
    stats_reset = stats.reset_index()
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for record in stats_reset.to_dict(orient='records'):
            f.write(json.dumps(record) + '\n')
    
    print(f"Statistics saved to: {output_file}")

def main():
    """
    Main function to run the script with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Calculate model accuracy statistics with token counting')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input JSONL file with all data')
    parser.add_argument('--output', type=str, default='analysis/stats.jsonl',
                        help='Path to the output JSONL file with statistics')
    parser.add_argument('--groupby', type=str, choices=['env', 'type', 'both'], default='both',
                        help='Group statistics by environment, type, or both')
    
    args = parser.parse_args()
    
    # Set up input and output files
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {input_file}")
    data = read_jsonl(input_file)
    print(f"Loaded {len(data)} records")
    
    # Calculate statistics
    groupby = None if args.groupby == 'both' else args.groupby
    print(f"Calculating statistics grouped by: {args.groupby}")
    stats = calculate_accuracy(data, groupby)
    
    # Save statistics to JSONL
    save_stats_to_jsonl(stats, output_file)
    
    # Print statistics
    print("\nAccuracy Statistics:")
    print(stats)

if __name__ == "__main__":
    main()