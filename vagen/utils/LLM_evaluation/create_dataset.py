import pandas as pd
import yaml
import json
import argparse
import re
from pathlib import Path

def load_templates(template_path):
    """
    Load YAML template file.
    
    Args:
        template_path (str): Path to the YAML template file.
    
    Returns:
        dict: Loaded template data.
    """
    with open(template_path, 'r') as file:
        return yaml.safe_load(file)

def resolve_template_reference(templates, reference):
    """
    Resolve a template reference like ${prompt_templates.default_env.grounding}.
    
    Args:
        templates (dict): The full templates dictionary.
        reference (str): The reference string.
    
    Returns:
        str: The resolved template.
    """
    # Extract the reference path
    match = re.match(r'\${(.*)}', reference)
    if not match:
        return reference
    
    path = match.group(1).split('.')
    
    # Navigate through the templates dict to find the referenced template
    current = templates
    for part in path:
        if part in current:
            current = current[part]
        else:
            raise ValueError(f"Referenced template part '{part}' not found in path: {path}")
    
    # Make sure we got a string
    if not isinstance(current, str):
        raise ValueError(f"Referenced template is not a string: {reference}")
    
    return current

def get_template(templates, env, type_name):
    """
    Get the specific template for an environment and type, resolving any references.
    
    Args:
        templates (dict): Loaded templates dictionary.
        env (str): Environment name.
        type_name (str): Type name (grounding or worldmodeling).
    
    Returns:
        str: The resolved template string.
    """
    try:
        # Navigate the template structure to find the specific template
        prompt_templates = templates.get('prompt_templates', templates)
        
        # Check if env exists in templates
        if env in prompt_templates:
            # Check if type exists in env templates
            env_templates = prompt_templates[env]
            if type_name in env_templates:
                template = env_templates[type_name]
                
                # Check if this is a reference and resolve it
                if isinstance(template, str) and template.startswith('${'):
                    template = resolve_template_reference(templates, template)
                
                return template
    except Exception as e:
        print(f"Error finding template: {e}")
    
    raise ValueError(f"Template for env={env}, type={type_name} not found in templates")

def process_sample(row, templates, template_type):
    """
    Process a single sample using the specified templates.
    
    Args:
        row (Series): DataFrame row containing sample data.
        templates (dict): Loaded templates dictionary.
        template_type (str): Identifier for the template (e.g., 'old' or 'new').
    
    Returns:
        dict: Processed sample.
    """
    # Extract data from row with new column names
    sample_id = row['Id']
    env = row['Task']
    type_name = row['Type']
    
    # Map different fields based on the type
    if type_name.lower() == 'grounding':
        # For grounding, state_information_dict is Gt_state and description is Predicted_state
        state_dict = row['Gt_state']
        description = row['Predicted_state']
    elif type_name.lower() == 'worldmodeling':
        # For worldmodeling, state_information_dict is still Gt_state but description is now Predicted_state
        state_dict = row['Gt_state']
        description = row['Predicted_state']
    else:
        raise ValueError(f"Unknown type: {type_name}")
        
    human_answer = row['Human_answer']
    
    # Get template
    template = get_template(templates, env, type_name)
    
    # Fill template
    prompt = template.replace('{state_information_dict}', state_dict).replace('{natural_language_description}', description)
    
    # Default value for max_tokens if present in template
    if '{max_tokens}' in prompt:
        prompt = prompt.replace('{max_tokens}', '1000')
    
    # Create sample
    sample = {
        "id": f"{sample_id}_{template_type}",  # Add template type to ID to make it unique
        "env": env,
        "type": type_name,
        "gt_state": state_dict,
        "predicted_state": description,
        "human_answer": human_answer,
        "prompt": prompt,
        "template_type": template_type  # Add template type to sample
    }
    
    return sample

def create_datasets(csv_path, old_templates_path, new_templates_path, output_dir):
    """
    Create datasets from raw CSV data using both old and new templates.
    
    Args:
        csv_path (str): Path to the raw CSV data.
        old_templates_path (str): Path to the old templates YAML file.
        new_templates_path (str): Path to the new templates YAML file.
        output_dir (str): Directory to save the output JSONL files.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Load templates
    old_templates = load_templates(old_templates_path)
    new_templates = load_templates(new_templates_path)
    
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    old_output_path = output_dir_path / "old_samples.jsonl"
    new_output_path = output_dir_path / "new_samples.jsonl"
    
    # Process with old templates
    with open(old_output_path, 'w') as old_file:
        for _, row in df.iterrows():
            try:
                sample = process_sample(row, old_templates, 'old')
                old_file.write(json.dumps(sample) + '\n')
            except Exception as e:
                print(f"Error processing row with old templates: {e}")
                print(f"Row data: {row}")
    
    print(f"Created dataset with old templates at {old_output_path}")
    
    # Process with new templates
    with open(new_output_path, 'w') as new_file:
        for _, row in df.iterrows():
            try:
                sample = process_sample(row, new_templates, 'new')
                new_file.write(json.dumps(sample) + '\n')
            except Exception as e:
                print(f"Error processing row with new templates: {e}")
                print(f"Row data: {row}")
    
    print(f"Created dataset with new templates at {new_output_path}")

def main():
    """
    Main function to run the script with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Create datasets from raw CSV and templates')
    parser.add_argument('--csv', type=str, default='raw_data/data.csv',
                        help='Path to the raw CSV data')
    parser.add_argument('--old_templates', type=str, default='raw_data/templates/old_prompts.yaml',
                        help='Path to the old templates YAML file')
    parser.add_argument('--new_templates', type=str, default='raw_data/templates/new_prompts.yaml',
                        help='Path to the new templates YAML file')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the output JSONL files')
    
    args = parser.parse_args()
    
    create_datasets(args.csv, args.old_templates, args.new_templates, args.output_dir)

if __name__ == "__main__":
    main()