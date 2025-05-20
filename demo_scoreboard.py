#!/usr/bin/env python3
"""
Demo script for the Scoreboard Recognition System
"""

import os
import json
import argparse
from scoreboard_recognition import ScoreboardRecognizer

def main():
    """Run a demo of the scoreboard recognition system"""
    parser = argparse.ArgumentParser(description="Demo for Scoreboard Recognition")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to an image file to process")
    parser.add_argument("--api-key", type=str,
                        help="Google API key (optional, will use GOOGLE_API_KEY environment variable if not set)")
    parser.add_argument("--examples-path", type=str, 
                       default="continued_iterations_results/best_examples/examples.json",
                       help="Path to the examples JSON file")
    parser.add_argument("--examples-dir", type=str, 
                       default="continued_iterations_results/best_examples",
                       help="Directory containing example images")
    args = parser.parse_args()
    
    # Validate that image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist")
        return 1
    
    print(f"Running scoreboard recognition demo on: {args.image}")
    print("Initializing recognizer...")
    
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
    
    # Process the image
    print("Processing image...")
    try:
        result = recognizer.extract_scoreboard(args.image)
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    # Display the results
    print("\n===== RECOGNITION RESULTS =====")
    
    if result.get("success", False):
        print("✅ Successfully extracted scoreboard information:")
        print(f"  Home Score: {result.get('home_score', 'N/A')}")
        print(f"  Away Score: {result.get('away_score', 'N/A')}")
        print(f"  Clock: {result.get('clock', 'N/A')}")
        print(f"  Period: {result.get('period', 'N/A')}")
        
        print("\nConfidence scores:")
        confidence = result.get("confidence", {})
        print(f"  Home Score: {confidence.get('home_score', 'N/A')}%")
        print(f"  Away Score: {confidence.get('away_score', 'N/A')}%")
        print(f"  Clock: {confidence.get('clock', 'N/A')}%")
        print(f"  Period: {confidence.get('period', 'N/A')}%")
        
        print(f"\nInference time: {result.get('inference_time', 0):.2f} seconds")
        
        if "warnings" in result:
            print(f"\n⚠️ Warnings: {result['warnings']}")
    else:
        print("❌ Failed to extract scoreboard information")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Save the result to a file
    output_file = f"{os.path.splitext(args.image)[0]}_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResult saved to: {output_file}")
    return 0

if __name__ == "__main__":
    exit(main()) 