import os
import pandas as pd
import glob
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

def analyze_results(env_name, model_name):
    """
    Analyze results for a specific environment and model.
    Calculate correct answer percentage for each test file.
    
    Args:
        env_name (str): Name of the environment folder.
        model_name (str): Name of the model folder (e.g., 'gpt', 'gemini').
    
    Returns:
        dict: Dictionary containing statistics for each file.
    """
    # Define analysis directory path
    analysis_dir = Path(f"analysis/{model_name}/{env_name}")
    
    if not analysis_dir.exists():
        print(f"Error: Directory {analysis_dir} not found.")
        return {}
    
    # Get all CSV files in the analysis directory
    csv_files = list(analysis_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in directory: {analysis_dir}")
        return {}
    
    results = {}
    
    for file_path in csv_files:
        file_name = file_path.stem
        print(f"Analyzing file: {file_name}")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Skip if file is empty or doesn't have required columns
        if df.empty or 'parsed_answer' not in df.columns:
            print(f"Warning: File {file_name} is empty or missing parsed_answer column")
            continue
        
        # Count total samples
        total_samples = len(df)
        valid_samples = df['parsed_answer'].notna().sum()
        
        if valid_samples == 0:
            print(f"Warning: No valid answers found in {file_name}")
            continue
        
        # Determine expected correct answer based on file name
        if "correct" in file_name.lower():
            expected_answer = "YES"
            correct_count = (df['parsed_answer'] == "YES").sum()
        else:  # "incorrect" in file_name
            expected_answer = "NO"
            correct_count = (df['parsed_answer'] == "NO").sum()
        
        # Calculate percentage
        correct_percentage = (correct_count / valid_samples) * 100 if valid_samples > 0 else 0
        
        # Store results
        results[file_name] = {
            'total': total_samples,
            'valid': valid_samples,
            'correct': correct_count,
            'percentage': correct_percentage,
            'expected_answer': expected_answer
        }
        
        print(f"  Total samples: {total_samples}")
        print(f"  Valid answers: {valid_samples}")
        print(f"  Correct answers ({expected_answer}): {correct_count}")
        print(f"  Correct percentage: {correct_percentage:.2f}%")
    
    return results

def create_summary_report(env_name, results_by_model):
    """
    Create a summary report with statistics for all models in the given environment.
    
    Args:
        env_name (str): Name of the environment folder.
        results_by_model (dict): Dictionary containing results for each model.
    """
    print("\n===== SUMMARY REPORT =====")
    print(f"Environment: {env_name}\n")
    
    # Create output directory for reports
    output_dir = Path(f"analysis/reports/{env_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV report
    report_data = []
    
    # Prepare data for bar chart
    models = list(results_by_model.keys())
    file_names = set()
    for model_results in results_by_model.values():
        file_names.update(model_results.keys())
    file_names = sorted(list(file_names))
    
    # Create a dictionary to store percentages for plotting
    plot_data = {file_name: [] for file_name in file_names}
    
    # Collect data for report and plotting
    for model in models:
        for file_name in file_names:
            if file_name in results_by_model[model]:
                result = results_by_model[model][file_name]
                
                # Add to report data
                report_data.append({
                    'Model': model,
                    'File': file_name,
                    'Total Samples': result['total'],
                    'Valid Answers': result['valid'],
                    'Correct Answers': result['correct'],
                    'Percentage': f"{result['percentage']:.2f}%",
                    'Expected Answer': result['expected_answer']
                })
                
                # Add to plot data
                plot_data[file_name].append(result['percentage'])
            else:
                # If model doesn't have results for this file
                plot_data[file_name].append(0)
    
    # Create CSV report
    report_df = pd.DataFrame(report_data)
    report_path = output_dir / f"{env_name}_summary_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"Summary report saved to: {report_path}")
    
    # Create bar chart visualization
    create_bar_chart(env_name, models, file_names, plot_data, output_dir)

def create_bar_chart(env_name, models, file_names, plot_data, output_dir):
    """
    Create a bar chart visualization of correct answer percentages.
    
    Args:
        env_name (str): Name of the environment.
        models (list): List of model names.
        file_names (list): List of file names.
        plot_data (dict): Dictionary with file names as keys and lists of percentages as values.
        output_dir (Path): Output directory for the chart.
    """
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Number of groups (files)
    n_groups = len(file_names)
    
    # Width of each bar
    bar_width = 0.8 / len(models)
    
    # Position of bars on x-axis
    index = np.arange(n_groups)
    
    # Colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create bars for each model
    for i, model in enumerate(models):
        # Extract data for this model
        model_data = [plot_data[file][i] if i < len(plot_data[file]) else 0 for file in file_names]
        
        # Create bars
        plt.bar(index + i * bar_width, model_data, bar_width,
                alpha=0.8, color=colors[i % len(colors)], label=model)
    
    # Add labels and title
    plt.xlabel('Test Files')
    plt.ylabel('Correct Answer Percentage (%)')
    plt.title(f'Model Performance on {env_name} Environment')
    plt.xticks(index + bar_width * (len(models) - 1) / 2, [fname.replace('_', '\n') for fname in file_names])
    plt.ylim(0, 105)  # Set y-axis limit to 0-105% for better visualization
    
    # Add grid lines for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    chart_path = output_dir / f"{env_name}_performance_chart.png"
    plt.savefig(chart_path)
    print(f"Performance chart saved to: {chart_path}")
    
    # Close the figure to free memory
    plt.close()

def main():
    """
    Main function to run the script with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze and compare model performances')
    parser.add_argument('env_name', type=str, help='Environment name (e.g., navigation)')
    parser.add_argument('--models', nargs='+', default=['gpt', 'gemini'], 
                        help='List of models to analyze (default: gpt gemini)')
    
    args = parser.parse_args()
    
    # Collect results for each model
    results_by_model = {}
    
    for model in args.models:
        print(f"\n===== Analyzing {model.upper()} results =====")
        results = analyze_results(args.env_name, model)
        
        if results:
            results_by_model[model] = results
        else:
            print(f"No results found for model: {model}")
    
    # Create summary report if we have results
    if results_by_model:
        create_summary_report(args.env_name, results_by_model)
    else:
        print("No results to generate report.")

if __name__ == "__main__":
    main()