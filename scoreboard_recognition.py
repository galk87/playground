#!/usr/bin/env python3
"""
Production-Ready Scoreboard Recognition System

This module provides functionality to extract scoreboard information from basketball
game images using the best prompts identified through iterative training.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Union, Optional, Tuple, Any
import logging
from pathlib import Path
import concurrent.futures

import google.generativeai as genai
import re
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scoreboard_recognition')

class ScoreboardRecognizer:
    """
    A class for recognizing scoreboard information from basketball game images.
    Uses the best examples/prompts from iterative training.
    """
    
    def __init__(
        self,
        examples_path: str = "continued_iterations_results/best_examples/examples.json",
        examples_dir: str = "continued_iterations_results/best_examples",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        confidence_threshold: float = 80.0,
        cache_results: bool = True,
        cache_dir: str = ".cache/scoreboard_results"
    ):
        """
        Initialize the ScoreboardRecognizer with the best examples.
        
        Args:
            examples_path: Path to the JSON file containing the best examples
            examples_dir: Directory containing example images
            api_key: Google API key for the Gemini model
            model_name: Gemini model name to use
            confidence_threshold: Minimum confidence score to accept a prediction
            cache_results: Whether to cache results to avoid redundant API calls
            cache_dir: Directory to store cache files
        """
        # Configure API
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Load examples
        self.examples_path = examples_path
        self.examples_dir = examples_dir
        self.examples = self._load_examples()
        
        # Settings
        self.confidence_threshold = confidence_threshold
        self.cache_results = cache_results
        
        # Setup cache
        if cache_results:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled. Results will be stored in {self.cache_dir}")
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load the best examples from the JSON file."""
        try:
            with open(self.examples_path, 'r') as f:
                examples = json.load(f)
            
            logger.info(f"Loaded {len(examples)} examples from {self.examples_path}")
            return examples
        except Exception as e:
            logger.error(f"Error loading examples: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load an image from the given path."""
        try:
            return Image.open(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from model response text."""
        # Look for JSON pattern in the response
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from marked section: {json_str[:100]}...")
        
        # Try to find JSON without markers
        try:
            # Try to load the entire response as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from entire response: {text[:100]}...")
            return None
    
    def _get_cache_path(self, image_path: str) -> Path:
        """Get the path to the cached result for an image."""
        image_hash = hash(os.path.abspath(image_path))
        return self.cache_dir / f"result_{image_hash}.json"
    
    def _check_cache(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Check if a result is cached for the given image."""
        if not self.cache_results:
            return None
        
        cache_path = self._get_cache_path(image_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    result = json.load(f)
                logger.debug(f"Found cached result for {image_path}")
                return result
            except Exception as e:
                logger.warning(f"Error loading cached result: {e}")
        
        return None
    
    def _save_to_cache(self, image_path: str, result: Dict[str, Any]) -> None:
        """Save a result to the cache."""
        if not self.cache_results:
            return
        
        cache_path = self._get_cache_path(image_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            logger.debug(f"Saved result to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def extract_scoreboard(self, image_path: str, return_raw_output: bool = False) -> Dict[str, Any]:
        """
        Extract scoreboard information from an image.
        
        Args:
            image_path: Path to the image file
            return_raw_output: Whether to include raw model output in the result
            
        Returns:
            Dictionary with scoreboard information and confidence scores
        """
        # Check cache first
        cached_result = self._check_cache(image_path)
        if cached_result:
            return cached_result
        
        # Create content for the prompt
        content = []
        
        # Add instructions
        content.append("You are an expert at analyzing basketball scoreboards.")
        content.append("Extract the following information: home_score, away_score, clock, and period.")
        content.append("Here are some examples:\n\n")
        
        # Add each example
        for i, example in enumerate(self.examples, 1):
            image_id = example["image_id"]
            example_image_path = os.path.join(self.examples_dir, f"example_{image_id}.jpg")
            
            if not os.path.exists(example_image_path):
                logger.warning(f"Example image {image_id} not found, skipping")
                continue
            
            # Add example
            content.append(f"Example #{i}\nInput: ")
            img = self._load_image(example_image_path)
            if img:
                content.append(img)
            else:
                continue
            
            # Add expected output
            content.append("\nOutput:\n```json\n")
            output = {
                "home_score": example["home_score"],
                "away_score": example["away_score"],
                "clock": example["clock"],
                "period": example["period"],
                "confidence": {
                    "home_score": 98,
                    "away_score": 98,
                    "clock": 96,
                    "period": 97
                }
            }
            content.append(json.dumps(output, indent=2))
            content.append("\n```\n\n")
        
        # Add test instructions
        content.append("\nNow analyze this new scoreboard image and extract the information in JSON format.\n")
        content.append("Input:")
        
        # Add test image
        test_img = self._load_image(image_path)
        if not test_img:
            return {
                "error": f"Could not load test image: {image_path}",
                "success": False
            }
        content.append(test_img)
        
        # Call the model
        start_time = time.time()
        try:
            response = self.model.generate_content(content)
            
            # Process the response
            if hasattr(response, 'text'):
                raw_response = response.text
            else:
                raw_response = str(response)
            
            # Extract JSON from response
            extraction = self._extract_json_from_response(raw_response)
            inference_time = time.time() - start_time
            
            # Create result object
            if extraction:
                result = {
                    "success": True,
                    "home_score": extraction.get("home_score"),
                    "away_score": extraction.get("away_score"),
                    "clock": extraction.get("clock"),
                    "period": extraction.get("period"),
                    "confidence": extraction.get("confidence", {}),
                    "inference_time": inference_time,
                }
                
                # Check if confidence is high enough
                confidences = extraction.get("confidence", {})
                low_confidence_fields = []
                
                for field in ["home_score", "away_score", "clock", "period"]:
                    confidence = confidences.get(field, 0)
                    if confidence < self.confidence_threshold:
                        low_confidence_fields.append(field)
                
                if low_confidence_fields:
                    result["warnings"] = f"Low confidence for fields: {', '.join(low_confidence_fields)}"
            else:
                result = {
                    "success": False,
                    "error": "Could not extract JSON from model response",
                    "inference_time": inference_time,
                }
            
            # Include raw output if requested
            if return_raw_output:
                result["raw_output"] = raw_response
            
            # Cache the result
            self._save_to_cache(image_path, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {
                "success": False,
                "error": str(e),
                "inference_time": time.time() - start_time
            }
    
    def batch_extract(
        self, 
        image_paths: List[str], 
        max_workers: int = 1,
        return_raw_output: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract scoreboard information from multiple images.
        
        Args:
            image_paths: List of paths to image files
            max_workers: Maximum number of parallel workers (1 for sequential processing)
            return_raw_output: Whether to include raw model output in results
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping image paths to extraction results
        """
        results = {}
        
        # For a single worker, process sequentially
        if max_workers == 1:
            iterator = tqdm(image_paths, desc="Processing images") if show_progress else image_paths
            for image_path in iterator:
                results[image_path] = self.extract_scoreboard(image_path, return_raw_output)
            return results
        
        # For multiple workers, use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all extraction tasks
            future_to_path = {
                executor.submit(self.extract_scoreboard, path, return_raw_output): path 
                for path in image_paths
            }
            
            # Process results as they complete
            iterator = concurrent.futures.as_completed(future_to_path)
            if show_progress:
                iterator = tqdm(iterator, total=len(image_paths), desc="Processing images")
                
            for future in iterator:
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path] = result
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results[path] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def extract_and_save_batch(
        self,
        image_dir: str,
        output_file: str,
        file_pattern: str = "*.jpg",
        max_workers: int = 1,
        show_progress: bool = True
    ) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
        """
        Extract scoreboard information from all images in a directory and save to a file.
        
        Args:
            image_dir: Directory containing images to process
            output_file: Path to save the results JSON file
            file_pattern: Pattern to match image files (*.jpg, *.png, etc.)
            max_workers: Maximum number of parallel workers
            show_progress: Whether to show a progress bar
            
        Returns:
            Tuple of (results dict, success count, total count)
        """
        # Find all matching images
        image_paths = []
        for pattern in file_pattern.split(","):
            pattern = pattern.strip()
            image_paths.extend(list(Path(image_dir).glob(pattern)))
        
        # Convert to strings
        image_paths = [str(p) for p in image_paths]
        logger.info(f"Found {len(image_paths)} images matching pattern {file_pattern} in {image_dir}")
        
        if not image_paths:
            logger.warning(f"No images found in {image_dir} matching {file_pattern}")
            return {}, 0, 0
        
        # Process all images
        results = self.batch_extract(
            image_paths,
            max_workers=max_workers,
            return_raw_output=False,
            show_progress=show_progress
        )
        
        # Count successes
        success_count = sum(1 for r in results.values() if r.get("success", False))
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processed {len(results)} images with {success_count} successes")
        logger.info(f"Results saved to {output_file}")
        
        return results, success_count, len(results)


def main():
    """Main function to run as a command-line tool."""
    parser = argparse.ArgumentParser(description="Extract scoreboard information from basketball images")
    
    # Main arguments
    parser.add_argument("--image", type=str, help="Path to a single image for processing")
    parser.add_argument("--image-dir", type=str, help="Directory containing images for batch processing")
    parser.add_argument("--output", type=str, default="scoreboard_results.json", 
                        help="Output file for results (default: scoreboard_results.json)")
    
    # Configuration arguments
    parser.add_argument("--examples-path", type=str, 
                        default="continued_iterations_results/best_examples/examples.json",
                        help="Path to examples JSON file")
    parser.add_argument("--examples-dir", type=str, 
                        default="continued_iterations_results/best_examples",
                        help="Directory containing example images")
    parser.add_argument("--api-key", type=str,
                        help="Google Gemini API key (will use GOOGLE_API_KEY env var if not provided)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Model name to use (default: gemini-2.0-flash)")
    parser.add_argument("--confidence-threshold", type=float, default=80.0,
                        help="Minimum confidence score to accept (default: 80.0)")
    
    # Batch processing arguments
    parser.add_argument("--file-pattern", type=str, default="*.jpg,*.png,*.jpeg",
                        help="File pattern(s) for batch processing (comma-separated, default: *.jpg,*.png,*.jpeg)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers for batch processing (default: 1)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching of results")
    parser.add_argument("--cache-dir", type=str, default=".cache/scoreboard_results",
                        help="Directory to store cached results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Initialize the recognizer
    recognizer = ScoreboardRecognizer(
        examples_path=args.examples_path,
        examples_dir=args.examples_dir,
        api_key=args.api_key,
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        cache_results=not args.no_cache,
        cache_dir=args.cache_dir
    )
    
    # Process based on arguments
    if args.image:
        # Single image processing
        logger.info(f"Processing single image: {args.image}")
        result = recognizer.extract_scoreboard(args.image, return_raw_output=args.verbose)
        
        # Save the result
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Display result
        if result.get("success", False):
            print(f"\nScoreboard information extracted successfully:")
            print(f"Home Score: {result.get('home_score', 'N/A')} ({result.get('confidence', {}).get('home_score', 'N/A')}%)")
            print(f"Away Score: {result.get('away_score', 'N/A')} ({result.get('confidence', {}).get('away_score', 'N/A')}%)")
            print(f"Clock: {result.get('clock', 'N/A')} ({result.get('confidence', {}).get('clock', 'N/A')}%)")
            print(f"Period: {result.get('period', 'N/A')} ({result.get('confidence', {}).get('period', 'N/A')}%)")
            print(f"Inference Time: {result.get('inference_time', 'N/A'):.2f}s")
            
            if "warnings" in result:
                print(f"Warnings: {result['warnings']}")
        else:
            print(f"\nFailed to extract scoreboard information:")
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"\nResult saved to {args.output}")
    
    elif args.image_dir:
        # Batch processing
        logger.info(f"Processing images in directory: {args.image_dir}")
        results, success_count, total_count = recognizer.extract_and_save_batch(
            image_dir=args.image_dir,
            output_file=args.output,
            file_pattern=args.file_pattern,
            max_workers=args.max_workers,
            show_progress=True
        )
        
        # Display summary
        print(f"\nBatch processing complete:")
        print(f"Processed {total_count} images")
        if total_count > 0:
            print(f"Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            print(f"Results saved to {args.output}")
        else:
            print("No images were processed.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 