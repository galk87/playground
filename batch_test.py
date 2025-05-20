#!/usr/bin/env python3
"""
Batch testing script for scoreboard recognition
This script allows testing specific image files for scoreboard recognition
"""

import os
import sys
import argparse
import json
from scoreboard_recognition import ScoreboardRecognizer
from tqdm import tqdm

def main():
    """Run batch testing on specific image files"""
    parser = argparse.ArgumentParser(description="Batch testing for scoreboard recognition")
    parser.add_argument("--image-files", type=str, nargs="+", required=True,
                        help="List of image files to process")
    parser.add_argument("--output", type=str, default="batch_results.json",
                        help="Output file for results")
    parser.add_argument("--api-key", type=str,
                        help="Google API key (optional, will use GOOGLE_API_KEY environment variable if not set)")
    parser.add_argument("--examples-path", type=str, 
                        default="accumulated_scoreboard_results/accumulated_examples/examples.json",
                        help="Path to the examples JSON file")
    parser.add_argument("--examples-dir", type=str, 
                        default="accumulated_scoreboard_results/accumulated_examples",
                        help="Directory containing example images")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    
    # Validate that files exist
    valid_files = []
    for file_path in args.image_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found, skipping: {file_path}")
    
    if not valid_files:
        print("Error: No valid image files to process")
        return 1
    
    print(f"Testing scoreboard recognition on {len(valid_files)} images")
    
    # Initialize the recognizer
    try:
        recognizer = ScoreboardRecognizer(
            examples_path=args.examples_path,
            examples_dir=args.examples_dir,
            api_key=args.api_key,
        )
    except Exception as e:
        print(f"Error initializing recognizer: {e}")
        return 1
    
    # Process all images
    print(f"Processing images with {args.max_workers} workers...")
    results = recognizer.batch_extract(
        valid_files,
        max_workers=args.max_workers,
        return_raw_output=args.verbose,
        show_progress=True
    )
    
    # Count successes
    success_count = sum(1 for r in results.values() if r.get("success", False))
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    print(f"\n===== BATCH PROCESSING RESULTS =====")
    print(f"Total images processed: {len(valid_files)}")
    print(f"Successful extractions: {success_count}")
    print(f"Success rate: {success_count/len(valid_files)*100:.1f}%")
    
    if args.verbose:
        # Display details for each result
        print("\nDetailed results:")
        for file_path, result in results.items():
            file_name = os.path.basename(file_path)
            if result.get("success", False):
                print(f"\n{file_name}:")
                print(f"  Home Score: {result.get('home_score')} ({result.get('confidence', {}).get('home_score')}%)")
                print(f"  Away Score: {result.get('away_score')} ({result.get('confidence', {}).get('away_score')}%)")
                print(f"  Clock: {result.get('clock')} ({result.get('confidence', {}).get('clock')}%)")
                print(f"  Period: {result.get('period')} ({result.get('confidence', {}).get('period')}%)")
            else:
                print(f"\n{file_name}: Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 