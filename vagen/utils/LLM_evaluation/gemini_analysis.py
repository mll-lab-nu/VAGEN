import os
import re
import pandas as pd
import time
from pathlib import Path
import argparse
import google.generativeai as genai

def extract_answer(response_text):
    """
    Extract YES or NO answer from the response text.
    
    Args:
        response_text (str): The response text from Gemini.
        
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

def get_gemini_response(model, prompt):
    """
    Get response from Gemini model for the given prompt.
    
    Args:
        model: The Gemini model instance.
        prompt (str): The input prompt to send to Gemini.
        
    Returns:
        str: The response from Gemini.
    """
    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Wait and retry on rate limit errors
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            print("Rate limit hit. Waiting for 20 seconds...")
            time.sleep(20)
            return get_gemini_response(model, prompt)
        return f"Error: {e}"

def process_csv_file(model, input_file, output_file):
    """
    Process a CSV file by sending each prompt to Gemini and saving the results.
    
    Args:
        model: The Gemini model instance.
        input_file (Path): Path to the input CSV file.
        output_file (Path): Path to the output CSV file.
    """
    print(f"Processing file: {input_file}")
    
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Add columns for Gemini response and parsed answer if they don't exist
    if 'response' not in df.columns:
        df['response'] = None
    if 'parsed_answer' not in df.columns:
        df['parsed_answer'] = None
    
    # Process each row
    for i, row in df.iterrows():
        prompt = row['prompt']
        
        # Add instruction to include answer in the required format
        formatted_prompt = (
            "You are a helpful assistant. When answering, include your final answer as YES or NO "
            "in the format <answer>YES</answer> or <answer>NO</answer>.\n\n" + prompt
        )
        
        print(f"Processing sample {i+1}/{len(df)}, id: {row['id']}")
        
        # Get response from Gemini
        response = get_gemini_response(model, formatted_prompt)
        
        # Extract answer from response
        parsed_answer = extract_answer(response)
        
        # Update the dataframe
        df.at[i, 'response'] = response
        df.at[i, 'parsed_answer'] = parsed_answer
        
        # Save progress after each sample to avoid losing data in case of errors
        df.to_csv(output_file, index=False)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    print(f"Completed processing file: {input_file}")
    print(f"Results saved to: {output_file}")

def analyze_env(env_name):
    """
    Analyze all CSV files in the specified environment folder using Gemini.
    
    Args:
        env_name (str): Name of the environment folder.
    """
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Check if API key is available
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Initialize the Gemini model
    model = genai.GenerativeModel(
        "gemini-1.5-flash",  # Use the specified model
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1500,
        }
    )
    
    # Define input and output paths
    input_dir = Path(f"data/{env_name}")
    output_dir = Path(f"analysis/gemini/{env_name}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files in the input directory (including subdirectories)
    input_files = list(input_dir.glob("**/*.csv"))
    
    if not input_files:
        print(f"No CSV files found in directory: {input_dir}")
        return
    
    print(f"Found {len(input_files)} CSV files to process.")
    
    # Process each CSV file
    for input_file in input_files:
        # Determine the relative path from the input_dir
        rel_path = input_file.relative_to(input_dir)
        
        # Create corresponding output file path
        output_file = output_dir / rel_path
        
        # Create parent directories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process the file
        process_csv_file(model, input_file, output_file)

def main():
    """
    Main function to run the script with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze prompts using Gemini and extract answers')
    parser.add_argument('env_name', type=str, help='Environment name (e.g., navigation)')
    
    args = parser.parse_args()
    
    analyze_env(args.env_name)

if __name__ == "__main__":
    main()