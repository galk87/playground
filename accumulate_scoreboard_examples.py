#!/usr/bin/env python3
"""
Script to run the iterative improvement process with accumulating examples
across iterations. This corrects the implementation to keep all examples 
from all iterations, not just the best iteration.
"""

import os
import json
import argparse
import shutil
import time
from improved_global_iterative_images import ImprovedGlobalIterativeImprover

def run_with_accumulating_examples(val_dir, train_dir, output_dir, max_iterations=5, api_key=None):
    """Run improved global iterative with accumulated examples"""
    
    print(f"Running iterative improvement with accumulated examples")
    print(f"Output directory: {output_dir}")
    
    # Create improver
    improver = ImprovedGlobalIterativeImprover(
        val_dir=val_dir,
        train_dir=train_dir,
        output_dir=output_dir,
        max_iterations=max_iterations,
        patience=max_iterations,  # Set patience to max_iterations to avoid early stopping
        num_examples=5,  # Start with 5 examples
        api_key=api_key,
    )
    
    # Run the improver
    best_iter, best_acc = improver.run()
    
    # Now fix the best examples directory to include ALL examples from all iterations
    all_examples = []
    seen_image_ids = set()
    
    # Collect examples from all iterations
    for iteration in range(1, max_iterations + 1):
        iter_examples_file = os.path.join(output_dir, "iterations", f"iteration_{iteration}", "examples.json")
        if os.path.exists(iter_examples_file):
            try:
                with open(iter_examples_file, 'r') as f:
                    examples = json.load(f)
                    
                # Add only new examples (not seen before)
                for example in examples:
                    if example["image_id"] not in seen_image_ids:
                        all_examples.append(example)
                        seen_image_ids.add(example["image_id"])
                        print(f"Added example {example['image_id']} from iteration {iteration}")
            except Exception as e:
                print(f"Error loading examples from iteration {iteration}: {e}")
    
    print(f"\nCollected {len(all_examples)} unique examples from all iterations")
    
    # Create accumulated examples directory
    accumulated_dir = os.path.join(output_dir, "accumulated_examples")
    os.makedirs(accumulated_dir, exist_ok=True)
    
    # Save all examples to this directory
    with open(os.path.join(accumulated_dir, "examples.json"), 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    # Copy images for all examples
    for example in all_examples:
        image_id = example["image_id"]
        
        # Find source image
        src_path = None
        for dir_path in [train_dir, val_dir]:
            for ext in ['.jpg', '.png']:
                test_path = os.path.join(dir_path, f"{image_id}{ext}")
                if os.path.exists(test_path):
                    src_path = test_path
                    break
            if src_path:
                break
        
        if src_path:
            dst_path = os.path.join(accumulated_dir, f"example_{image_id}.jpg")
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied image for example {image_id}")
            except Exception as e:
                print(f"Error copying image for example {image_id}: {e}")
        else:
            print(f"Could not find image for example {image_id}")
    
    # Create a README for the accumulated examples
    with open(os.path.join(accumulated_dir, "README.md"), 'w') as f:
        f.write("""# Accumulated Examples for Scoreboard Recognition

This directory contains all examples collected across all iterations of the training process.

## Contents:
- `examples.json`: Metadata for all accumulated examples
- `example_*.jpg`: Example images

## How to use:
See the main README.md for instructions on using these examples with the 
scoreboard recognition system.
""")
    
    print(f"\nSaved accumulated examples to {accumulated_dir}")
    print(f"Best iteration: {best_iter}")
    print(f"Best accuracy: {best_acc:.2f}%")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Run iterative improvement with accumulating examples")
    parser.add_argument("--val-dir", type=str, default="combined_valid_dataset",
                      help="Directory with validation images")
    parser.add_argument("--train-dir", type=str, default="unified_dataset/train", 
                      help="Directory with training images")
    parser.add_argument("--output-dir", type=str, default="accumulated_examples_results",
                      help="Directory to save results")
    parser.add_argument("--max-iterations", type=int, default=5,
                      help="Maximum number of iterations")
    parser.add_argument("--api-key", type=str,
                      help="Google API key (optional, will use GOOGLE_API_KEY env var if not set)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No API key provided. Set --api-key or GOOGLE_API_KEY environment variable.")
        return 1
    
    # Run with accumulating examples
    return run_with_accumulating_examples(
        val_dir=args.val_dir,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        api_key=api_key
    )

if __name__ == "__main__":
    exit(main()) 