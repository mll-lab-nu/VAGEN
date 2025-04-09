#!/usr/bin/env python
# sequential_benchmark.py - Benchmark reward server with sequential interaction

import os
import sys
import time
import random
import argparse
import requests
import statistics
import threading
import concurrent.futures
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Adjust path to import necessary modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Import SVG environment components for testing
    from vagen.env.svgdino.config import SVGDINOConfig
    from vagen.env.svgdino.svg_utils import process_and_rasterize_svg, load_svg_dataset, is_valid_svg
    from datasets import load_dataset
    VAGEN_AVAILABLE = True
except ImportError:
    print("Warning: vagen modules could not be imported. Make sure they're in your PYTHONPATH.")
    VAGEN_AVAILABLE = False
    sys.exit(1)

class InteractionStats:
    """Track statistics for each interaction step"""
    def __init__(self):
        self.request_times = []
        self.success_count = 0
        self.error_count = 0
    
    def add_request_time(self, elapsed_time, success=True):
        self.request_times.append(elapsed_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_stats(self):
        if not self.request_times:
            return {
                "min_time": 0,
                "max_time": 0,
                "avg_time": 0,
                "median_time": 0,
                "success_rate": 0,
                "total_requests": 0
            }
        
        return {
            "min_time": min(self.request_times),
            "max_time": max(self.request_times),
            "avg_time": statistics.mean(self.request_times),
            "median_time": statistics.median(self.request_times),
            "percentile_95": statistics.quantiles(self.request_times, n=20)[18] if len(self.request_times) >= 20 else max(self.request_times),
            "success_rate": (self.success_count / (self.success_count + self.error_count)) * 100 if (self.success_count + self.error_count) > 0 else 0,
            "total_requests": self.success_count + self.error_count
        }

class BenchmarkStats:
    """Class to track benchmark statistics for all interactions"""
    
    def __init__(self, num_interactions):
        self.interaction_stats = [InteractionStats() for _ in range(num_interactions)]
        self.overall_times = []  # Total time for all interactions
        self.success_count = 0
        self.error_count = 0
        self.lock = threading.Lock()
    
    def add_interaction_result(self, interaction_idx, elapsed_time, success=True):
        with self.lock:
            self.interaction_stats[interaction_idx].add_request_time(elapsed_time, success)
    
    def add_overall_time(self, total_time, success=True):
        with self.lock:
            self.overall_times.append(total_time)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
    
    def get_stats(self):
        interaction_stats = [stat.get_stats() for stat in self.interaction_stats]
        
        if not self.overall_times:
            overall_stats = {
                "min_time": 0,
                "max_time": 0,
                "avg_time": 0,
                "median_time": 0,
                "success_rate": 0,
                "total_requests": 0
            }
        else:
            overall_stats = {
                "min_time": min(self.overall_times),
                "max_time": max(self.overall_times),
                "avg_time": statistics.mean(self.overall_times),
                "median_time": statistics.median(self.overall_times),
                "percentile_95": statistics.quantiles(self.overall_times, n=20)[18] if len(self.overall_times) >= 20 else max(self.overall_times),
                "success_rate": (self.success_count / (self.success_count + self.error_count)) * 100 if (self.success_count + self.error_count) > 0 else 0,
                "total_requests": self.success_count + self.error_count
            }
        
        return {
            "overall": overall_stats,
            "interactions": interaction_stats
        }

def make_reward_request(server_url, gt_svg, gen_svg):
    """Make a single request to the reward server and return timing and result"""
    start_time = time.time()
    success = False
    
    try:
        response = requests.post(
            f"{server_url}/compute_score",
            json={
                "gt_svg_code": gt_svg,
                "gen_svg_code": gen_svg
            },
            headers={"Content-Type": "application/json"},
            timeout=30  # 30-second timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            success = True
            result = response.json()
        else:
            result = {"error": f"Server returned status code {response.status_code}"}
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        result = {"error": str(e)}
        
    return success, elapsed_time, result

def modify_svg(svg_code, iteration):
    """Make increasingly significant modifications to the SVG based on iteration number"""
    lines = svg_code.split('\n')
    
    if len(lines) <= 3:
        return svg_code  # Not enough lines to modify
    
    # Apply modifications based on iteration
    # More iterations = more changes
    num_changes = iteration + 1
    changes_made = 0
    
    for _ in range(min(num_changes, len(lines) - 2)):
        # Select a random line to modify
        mod_line_idx = random.randint(1, len(lines) - 2)
        line = lines[mod_line_idx]
        
        # Change fill color if present
        if 'fill="' in line:
            colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink"]
            for color in colors:
                if f'fill="{color}"' in line:
                    new_color = random.choice([c for c in colors if c != color])
                    lines[mod_line_idx] = line.replace(f'fill="{color}"', f'fill="{new_color}"')
                    changes_made += 1
                    break
        
        # Or change a numeric value if present
        elif any(f'="{n}"' in line for n in map(str, range(10, 100))):
            for n in range(10, 100):
                if f'="{n}"' in line:
                    # Larger changes in later iterations
                    change_amount = random.randint(-5 * (iteration + 1), 5 * (iteration + 1))
                    new_val = max(1, n + change_amount)
                    lines[mod_line_idx] = line.replace(f'="{n}"', f'="{new_val}"')
                    changes_made += 1
                    break
        
        if changes_made >= num_changes:
            break
    
    return '\n'.join(lines)

def simulate_env_sequential_interactions(server_url, stats, dataset, thread_id, num_interactions=3):
    """Simulate a sequence of interactions between an environment and the reward server"""
    try:
        # Start timing the overall sequence
        overall_start_time = time.time()
        overall_success = True
        
        # Get a random sample from the dataset
        dataset_length = len(dataset)
        index = random.randint(0, dataset_length - 1)
        sample = dataset[index]
        
        # Extract SVG code (field names may vary depending on dataset structure)
        gt_svg_code = sample.get('Svg', sample.get('svg', ''))
        
        if not gt_svg_code or not is_valid_svg(gt_svg_code):
            print(f"Thread {thread_id}: Invalid ground truth SVG")
            stats.add_overall_time(0, False)
            return False
        
        # Perform multiple sequential interactions
        current_svg = gt_svg_code
        
        for i in range(num_interactions):
            # Create a modified version of the SVG for this interaction
            # Each iteration makes more significant changes
            gen_svg_code = modify_svg(current_svg, i)
            
            if not is_valid_svg(gen_svg_code):
                print(f"Thread {thread_id}, Interaction {i}: Invalid generated SVG")
                stats.add_interaction_result(i, 0, False)
                overall_success = False
                continue
            
            # Make request to the reward server
            success, elapsed_time, result = make_reward_request(server_url, gt_svg_code, gen_svg_code)
            
            # Record stats for this interaction
            stats.add_interaction_result(i, elapsed_time, success)
            
            if not success:
                overall_success = False
                # Continue to next interaction anyway
            else:
                # Update the current SVG for the next iteration
                # Simulate the model generating a better SVG based on reward
                current_svg = gen_svg_code
        
        # Record overall time for the entire sequence
        overall_elapsed_time = time.time() - overall_start_time
        stats.add_overall_time(overall_elapsed_time, overall_success)
        
        return overall_success
    
    except Exception as e:
        print(f"Thread {thread_id} error: {str(e)}")
        stats.add_overall_time(0, False)
        return False

def worker(server_url, stats, dataset, thread_id, num_interactions):
    """Worker function for concurrent requests"""
    return simulate_env_sequential_interactions(server_url, stats, dataset, thread_id, num_interactions)

def load_dataset_for_benchmark(dataset_name="starvector/svg-emoji-simple", 
                              data_dir="vagen/env/svg/data", 
                              split="train"):
    """Load the dataset for benchmarking"""
    print(f"Loading dataset {dataset_name} ({split} split)...")
    try:
        dataset = load_svg_dataset(data_dir, dataset_name, split)
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def run_benchmark(server_url, concurrency, num_environments, dataset, num_interactions=3):
    """Run the benchmark with specified concurrency using the real dataset"""
    print(f"Starting benchmark with {concurrency} concurrent environments")
    print(f"Each environment will perform {num_interactions} sequential interactions")
    print(f"Total number of environments to test: {num_environments}")
    
    # Check if server is up
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"Server health check failed with status code: {health_response.status_code}")
            return None
    except Exception as e:
        print(f"Server health check failed: {str(e)}")
        return None
    
    print("Server is up and running. Starting benchmark...")
    
    stats = BenchmarkStats(num_interactions)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit environments in batches to maintain concurrency
        futures = []
        for i in range(num_environments):
            futures.append(executor.submit(
                worker, server_url, stats, dataset, i, num_interactions
            ))
        
        # Show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
    
    return stats.get_stats()

def main():
    parser = argparse.ArgumentParser(description='Sequential Interaction Benchmark for SVG Reward Server')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:5000',
                        help='URL of the reward server')
    parser.add_argument('--concurrency', type=int, default=128,
                        help='Number of concurrent environments')
    parser.add_argument('--environments', type=int, default=256,
                        help='Total number of environments to test')
    parser.add_argument('--interactions', type=int, default=3,
                        help='Number of sequential interactions per environment')
    parser.add_argument('--dataset', type=str, default='starvector/svg-emoji-simple',
                        help='HuggingFace dataset name')
    parser.add_argument('--data-dir', type=str, default='vagen/env/svg/data',
                        help='Data directory')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    
    args = parser.parse_args()
    
    # Display benchmark configuration
    print("Sequential Interaction SVG Reward Server Benchmark")
    print("=================================================")
    print(f"Server URL: {args.url}")
    print(f"Concurrent Environments: {args.concurrency}")
    print(f"Total Environments: {args.environments}")
    print(f"Sequential Interactions: {args.interactions}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    print(f"Data Directory: {args.data_dir}")
    print("=================================================")
    
    # Load the dataset
    dataset = load_dataset_for_benchmark(args.dataset, args.data_dir, args.split)
    
    # Run the benchmark
    results = run_benchmark(args.url, args.concurrency, args.environments, dataset, args.interactions)
    
    if results:
        # Overall statistics
        overall = results["overall"]
        print("\nOverall Benchmark Results (Full Sequence):")
        print("==========================================")
        print(f"Total Environment Runs: {overall['total_requests']}")
        print(f"Success Rate: {overall['success_rate']:.2f}%")
        print(f"Average Full Sequence Time: {overall['avg_time']:.4f} seconds")
        print(f"Median Full Sequence Time: {overall['median_time']:.4f} seconds")
        print(f"95th Percentile: {overall['percentile_95']:.4f} seconds")
        print(f"Minimum Full Sequence Time: {overall['min_time']:.4f} seconds")
        print(f"Maximum Full Sequence Time: {overall['max_time']:.4f} seconds")
        
        # Individual interaction statistics
        print("\nIndividual Interaction Statistics:")
        print("===================================")
        for i, interaction in enumerate(results["interactions"]):
            print(f"\nInteraction {i+1}:")
            print(f"  Success Rate: {interaction['success_rate']:.2f}%")
            print(f"  Average Time: {interaction['avg_time']:.4f} seconds")
            print(f"  Median Time: {interaction['median_time']:.4f} seconds")
            print(f"  95th Percentile: {interaction['percentile_95']:.4f} seconds")
        
        # Calculate throughput
        throughput = args.concurrency / overall['avg_time'] * args.interactions
        
        print("\nPerformance Metrics:")
        print("===================")
        print(f"Estimated Throughput: {throughput:.2f} requests/second")
        print(f"Average Time Per Request: {(overall['avg_time'] / args.interactions):.4f} seconds")
        
        # Display recommendations
        print("\nRecommendations:")
        print("===============")
        if overall['avg_time'] / args.interactions > 0.5:
            print("- The server response time is high. Consider optimizing the score calculation.")
        if overall['success_rate'] < 95:
            print("- The success rate is low. Check for server errors or timeouts.")
        if overall['max_time'] / overall['min_time'] > 5:
            print("- There's high variance in response times. The server might be overloaded.")
            
        for i, interaction in enumerate(results["interactions"]):
            if i > 0 and interaction['avg_time'] > results["interactions"][i-1]['avg_time'] * 1.5:
                print(f"- Interaction {i+1} is significantly slower than previous interactions. " 
                      f"Check if server performance degrades over time.")
    else:
        print("Benchmark failed")

if __name__ == '__main__':
    main()