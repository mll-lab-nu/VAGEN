import os
import re
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import threading

output_lock = threading.Lock()

def extract_answer(response_text):
    """
    Extract YES or NO answer from the response text.
    
    Args:
        response_text (str): The response text from GPT.
        
    Returns:
        str: Extracted answer ('YES', 'NO', or None if not found).
    """
    # Look for <answer>YES</answer> or <answer>NO</answer> pattern
    answer_pattern = re.compile(r'<answer>(YES|NO)</answer>', re.IGNORECASE)
    match = answer_pattern.search(response_text)
    
    if match:
        return match.group(1).upper()  # Return YES or NO in uppercase
    else:
        print("Warning: Could not extract answer from response.")
        return None

def get_gpt_response(client, prompt, max_tokens=1500):
    """
    Get response from GPT model for the given prompt.
    
    Args:
        client (OpenAI): The OpenAI client instance.
        prompt (str): The input prompt to send to GPT.
        max_tokens (int): Maximum number of tokens to generate in the response.
        
    Returns:
        str: The response from GPT.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        # Wait and retry on rate limit errors
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            print("Rate limit hit. Waiting for 20 seconds...")
            time.sleep(20)
            return get_gpt_response(client, prompt, max_tokens)
        return f"Error: {e}"

def read_jsonl(file_path):
    """
    Read samples from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file.
    
    Returns:
        list: List of sample dictionaries.
    """
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                samples.append(json.loads(line.strip()))
    return samples

def process_sample(client, sample, output_path, max_tokens):
    """
    Process a single sample using GPT and save result.
    
    Args:
        client (OpenAI): The OpenAI client instance.
        sample (dict): The sample to process.
        output_path (str): Path to save the results JSONL file.
        max_tokens (int): Maximum number of tokens to generate in the response.
    """
    sample_id = sample["id"]
    prompt = sample["prompt"]
    env = sample["env"]
    type_name = sample["type"]
    gt_state = sample.get("gt_state", "")
    predicted_state = sample.get("predicted_state", "")
    human_answer = sample.get("human_answer", "")
    
    print(f"Processing sample id: {sample_id}")
    
    # Get response from GPT
    response = get_gpt_response(client, prompt, max_tokens)
    
    # Extract answer from response
    parsed_answer = extract_answer(response)
    
    # Create comprehensive result with consistent field order
    result = {
        "model": "gpt-4.1-nano",
        "id": sample_id,
        "env": env,
        "type": type_name,
        "human_answer": human_answer,
        "parsed_answer": parsed_answer,
        "gt_state": gt_state,
        "predicted_state": predicted_state,
        "response": response
    }
    
    # Write result to output file - use lock to ensure thread safety
    with output_lock:
        with open(output_path, 'a') as outfile:
            outfile.write(json.dumps(result) + '\n')
    
    return sample_id

def analyze_samples(samples_path, output_path, model_name="gpt-4.1-nano", max_parallel=8, max_tokens=1500):
    """
    Analyze samples using GPT and save results using parallel processing.
    
    Args:
        samples_path (str): Path to the samples JSONL file.
        output_path (str): Path to save the results JSONL file.
        model_name (str): GPT model to use.
        max_parallel (int): Maximum number of parallel requests.
        max_tokens (int): Maximum number of tokens to generate in the response.
    """
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Check if API key is available
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Read samples
    samples = read_jsonl(samples_path)
    
    # Clear output file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted existing output file: {output_path}")
    
    print(f"Processing {len(samples)} samples")
    print(f"Using parallel processing with max {max_parallel} workers")
    print(f"Using max_tokens: {max_tokens}")
    
    # Process samples in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all tasks
        futures = [executor.submit(process_sample, client, sample, output_path, max_tokens) 
                  for sample in samples]
        
        # Wait for all futures to complete
        for future in futures:
            try:
                sample_id = future.result()
                print(f"Completed processing sample id: {sample_id}")
            except Exception as e:
                print(f"Error processing sample: {e}")
    
    print(f"Completed processing all samples. Results saved to: {output_path}")

def main():
    """
    Main function to run the script with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze samples using GPT with parallel processing')
    parser.add_argument('--samples', type=str, default='data/samples.jsonl',
                        help='Path to the samples JSONL file')
    parser.add_argument('--output', type=str, default='analysis/gpt_results.jsonl',
                        help='Path to save the results JSONL file')
    parser.add_argument('--model', type=str, default='gpt-4.1-nano',
                        help='GPT model to use')
    parser.add_argument('--model_name', type=str, default='gpt-4.1-nano',
                        help='Name to use for the model in the output')
    parser.add_argument('--max_parallel', type=int, default=8,
                        help='Maximum number of parallel requests')
    parser.add_argument('--max_tokens', type=int, default=500,
                        help='Maximum number of tokens to generate in the response')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_samples(args.samples, args.output, args.model, args.max_parallel, args.max_tokens)

if __name__ == "__main__":
    main()