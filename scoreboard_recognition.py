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
import cv2 # Add OpenCV import
import numpy as np # Add numpy import for image conversion

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
        examples_path: str = "accumulated_scoreboard_results/accumulated_examples/examples.json",
        examples_dir: str = "accumulated_scoreboard_results/accumulated_examples",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        confidence_threshold: float = 80.0,
        cache_results: bool = True,
        cache_dir: str = ".cache/scoreboard_results",
        use_images: bool = True,
        dynamic_example_image_path: Optional[str] = None,
        dynamic_example_calibration_path: Optional[str] = None,
        dynamic_example_data: Optional[Dict[str, Any]] = None,
        dynamic_example_custom_description: Optional[str] = None
    ):
        """
        Initialize the ScoreboardRecognizer.
        Can optionally include a single dynamic few-shot example with calibration.
        
        Args:
            examples_path: Path to the JSON file containing all accumulated examples
            examples_dir: Directory containing example images
            api_key: Google API key for the Gemini model
            model_name: Gemini model name to use
            confidence_threshold: Minimum confidence score to accept a prediction
            cache_results: Whether to cache results to avoid redundant API calls
            cache_dir: Directory to store cache files
            use_images: Whether to include images in the examples (False for text-only mode)
            dynamic_example_image_path: Path to an image for a dynamic example
            dynamic_example_calibration_path: Path to a calibration file for a dynamic example
            dynamic_example_data: Data for a dynamic example
            dynamic_example_custom_description: A specific textual description of the dynamic example's layout.
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
        self.use_images = use_images
        
        self.dynamic_example_image_path = dynamic_example_image_path
        self.dynamic_example_calibration_path = dynamic_example_calibration_path
        self.dynamic_example_data = dynamic_example_data
        self.dynamic_example_custom_description = dynamic_example_custom_description
        
        # Setup cache
        if cache_results:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled. Results will be stored in {self.cache_dir}")
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load the best examples from the JSON file."""
        try:
            with open(self.examples_path, 'r') as f:
                examples_data = json.load(f)
            
            # Ensure examples are in the expected list format, even if JSON is a dict {examples: [...]}
            if isinstance(examples_data, dict) and 'examples' in examples_data:
                examples = examples_data['examples']
            elif isinstance(examples_data, list):
                examples = examples_data
            else:
                raise ValueError("Examples JSON is not in expected list format or dict with 'examples' key.")

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
    
    def _parse_calibration_data(self, calibration_file_path: str) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
        """Parse a protobuf-like calibration file and extract ROIs."""
        if not os.path.exists(calibration_file_path):
            logger.warning(f"Calibration file not found: {calibration_file_path}")
            return None
        rois = {}
        try:
            with open(calibration_file_path, 'r') as f:
                lines = f.readlines()

            current_object_lines = []
            brace_level = 0
            capturing_object = False

            for line in lines:
                stripped_line = line.strip()
                if stripped_line == "objects {":
                    if not capturing_object:
                        capturing_object = True
                        current_object_lines = [line]
                        brace_level = 1
                    else:
                        current_object_lines.append(line)
                        brace_level += line.count('{')
                        brace_level -= line.count('}')
                elif capturing_object:
                    current_object_lines.append(line)
                    brace_level += line.count('{')
                    brace_level -= line.count('}')
                    if brace_level == 0 and stripped_line == "}":
                        block = "".join(current_object_lines)
                        name_match = re.search(r'name: "(.*?)"', block)
                        if name_match:
                            name = name_match.group(1)
                            corners_match = re.findall(r'(leftUp|rightUp|rightDown|leftDown) \{\s*x: (\d+)\s*y: (\d+)\s*\}', block)
                            if len(corners_match) == 4:
                                xs = [int(x_val) for _, x_val, _ in corners_match]
                                ys = [int(y_val) for _, _, y_val in corners_match]
                                if xs and ys:
                                    x_min, x_max = min(xs), max(xs)
                                    y_min, y_max = min(ys), max(ys)
                                    w = x_max - x_min
                                    h = y_max - y_min
                                    if w > 0 and h > 0:
                                        rois[name] = (x_min, y_min, w, h)
                        capturing_object = False
                        current_object_lines = []
            return rois
        except Exception as e:
            logger.error(f"Error parsing calibration file {calibration_file_path}: {e}")
            return None
    
    def _draw_rois_on_image(self, image_pil: Image.Image, rois: Dict[str, Tuple[int, int, int, int]], color=(0, 255, 0), thickness=2) -> Image.Image:
        """Draws ROIs on a PIL image and returns a new PIL image."""
        if not rois:
            return image_pil
        
        # Convert PIL Image to OpenCV format
        image_cv = np.array(image_pil.convert('RGB')) 
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        for name, (x, y, w, h) in rois.items():
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, thickness)
            # ROI names are intentionally NOT drawn here to avoid clutter, as per user request

        # Convert OpenCV image back to PIL Image
        image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_cv_rgb)

    def extract_scoreboard(
        self, 
        image_path: str, 
        return_raw_output: bool = False, 
        custom_prompt_for_current_image: Optional[str] = None,
        prev_frame_guide_image: Optional[Image.Image] = None,
        prev_frame_guide_data: Optional[Dict[str, Any]] = None,
        prev_frame_guide_context_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        cached_result = self._check_cache(image_path)
        if cached_result:
            return cached_result
        
        content = []
        content.append("You are an expert at analyzing basketball scoreboards.")
        content.append("Extract the following information: home_score, away_score, clock, and period.")
        content.append("Important: The information for home_score, away_score, clock, and period should come from distinct, non-overlapping regions of the scoreboard. If your interpretation suggests that the visual source for two different fields (like home_score and period) is the same or significantly overlapping, please re-evaluate your extraction to ensure each field is unique and from its correct, separate area.")
        content.append("For some examples, I might provide calibration data (textually or drawn on the image) specifying regions of interest (ROIs), and a general description of the scoreboard layout. Pay close attention to these hints.")
        # content.append("Here are some examples from my knowledge base:\n\n") # Moved later

        # NEW: Previous Frame Context / Guide (inserted early for high priority)
        if prev_frame_guide_image and prev_frame_guide_data:
            content.append("\n--- Context from Immediately Preceding Frame ---\n")
            if prev_frame_guide_context_prompt:
                content.append(prev_frame_guide_context_prompt + "\n")
            content.append("This was the scoreboard in the immediately preceding moment. Scores and clock usually change incrementally. Significant deviations from this previous state are unlikely but possible (e.g., end of period, major score update). Use this as a strong reference.\n")
            content.append("Input (Previous Frame Image):\n")
            content.append(prev_frame_guide_image)
            content.append("\nOutput (Previously Extracted Data for that Frame):\n```json\n")
            # Ensure guide data is serializable and clean for the prompt
            clean_guide_data = {
                "home_score": prev_frame_guide_data.get("home_score"),
                "away_score": prev_frame_guide_data.get("away_score"),
                "clock": prev_frame_guide_data.get("clock"),
                "period": prev_frame_guide_data.get("period")
            }
            content.append(json.dumps(clean_guide_data, indent=2))
            content.append("\n```\n--- End of Preceding Frame Context ---\n\n")
            content.append("Now, consider this previous state when analyzing the new frame.\n")
        
        content.append("Here are some general examples from my knowledge base to further guide you:\n\n") # Original location, rephrased slightly
        
        # 1. Add existing examples from examples.json FIRST
        example_counter = 0
        for i, example in enumerate(self.examples, 1):
            example_counter = i
            example_info = f"Example #{example_counter}\n"
            calibration_text_for_json_example = ""
            if example.get("calibration_file_path"):
                calib_path_str = example["calibration_file_path"]
                calib_path = Path(calib_path_str)
                if not calib_path.is_absolute() and not calib_path.exists():
                    calib_path = Path.cwd() / calib_path_str
                if calib_path.exists():
                    parsed_rois_json = self._parse_calibration_data(str(calib_path))
                    if parsed_rois_json:
                        calibration_text_for_json_example = "Calibration data for this example (textual ROIs):\n"
                        for name, roi_coords in parsed_rois_json.items():
                            calibration_text_for_json_example += f"- {name}: {roi_coords}\n"
                        calibration_text_for_json_example += "\n"
                else:
                    logger.warning(f"JSON Example: Calib file {calib_path_str} not found.")

            content.append(example_info)
            if self.use_images:
                image_id = example.get("image_id", f"unknown_id_{i}")
                # Simplified path resolution for brevity in this edit - assume robust logic from previous edits is here
                example_image_path_str = str(Path(self.examples_dir) / f"example_{image_id}.jpg") 
                # Add more robust path discovery here as per your previous setup if needed
                img = self._load_image(example_image_path_str) 
                content.append("Input:")
                if img:
                    content.append(img)
                    content.append("\n" + calibration_text_for_json_example)
                else:
                    logger.warning(f"JSON Example: Could not load image {example_image_path_str}. Appending placeholder.")
                    content.append("[Image for example not loaded]\n" + calibration_text_for_json_example)
            else:
                content.append(f"Input: A basketball scoreboard (text description)...\n" + calibration_text_for_json_example)
            
            content.append("\nOutput:\n```json\n")
            output = {
                "home_score": example.get("home_score"), "away_score": example.get("away_score"),
                "clock": example.get("clock"), "period": example.get("period"),
                "confidence": example.get("confidence", {"home_score":98, "away_score":98, "clock":96, "period":97})
            }
            content.append(json.dumps(output, indent=2))
            content.append("\n```\n\n")

        # 2. Add dynamic example (as a special guide for the CURRENT image) LAST, before the actual test image
        # This block will construct the dynamic example part of the prompt
        dynamic_example_prepared_successfully = False
        if self.dynamic_example_image_path and self.dynamic_example_data:
            logger.info(f"Preparing dynamic guide example from: {self.dynamic_example_image_path}")
            
            dyn_img_pil = self._load_image(self.dynamic_example_image_path)
            dyn_img_pil_with_rois = None # This will hold the image with ROIs drawn
            textual_roi_list_for_dynamic = ""

            if dyn_img_pil:
                if self.dynamic_example_calibration_path:
                    parsed_rois_for_dynamic = self._parse_calibration_data(self.dynamic_example_calibration_path)
                    if parsed_rois_for_dynamic:
                        logger.info(f"Drawing {len(parsed_rois_for_dynamic)} ROIs and generating textual list for dynamic guide image.")
                        dyn_img_pil_with_rois = self._draw_rois_on_image(dyn_img_pil.copy(), parsed_rois_for_dynamic, color=(0,255,0), thickness=2)
                        
                        textual_roi_list_for_dynamic = "Key ROI Locations (drawn on image and listed as x, y, width, height):\n"
                        for name, roi_coords in parsed_rois_for_dynamic.items():
                            textual_roi_list_for_dynamic += f"- {name}: {roi_coords}\n"
                        textual_roi_list_for_dynamic += "\n"
                    else:
                        logger.warning(f"Could not parse ROIs from {self.dynamic_example_calibration_path} for dynamic guide. Image will be used as-is.")
                        dyn_img_pil_with_rois = dyn_img_pil # Use original image if ROIs can't be parsed
                else:
                    dyn_img_pil_with_rois = dyn_img_pil # Use original image if no calibration path given

                # If image loaded (and potentially drawn on), add this special block to content
                if dyn_img_pil_with_rois:
                    content.append("\n--- Detailed Guide for Current Scoreboard Analysis ---\n")
                    content.append("You will now analyze a new scoreboard. The following example provides specific details for an image similar to the one you need to process, including drawn ROIs and textual descriptions to guide your extraction.\n")
                    
                    current_dynamic_prompt_parts = []
                    if self.dynamic_example_custom_description:
                        current_dynamic_prompt_parts.append(f"Custom Description of Guide Image Layout:\n{self.dynamic_example_custom_description}\n")
                    
                    if textual_roi_list_for_dynamic:
                        current_dynamic_prompt_parts.append(textual_roi_list_for_dynamic)
                    
                    current_dynamic_prompt_parts.append("Guide Image with Highlighted ROIs (if any were specified):\n")
                    
                    content.append("Input Guide Details:\n" + "\n".join(current_dynamic_prompt_parts))
                    content.append(dyn_img_pil_with_rois) # Add the (potentially modified) guide image
                    
                    content.append("\nExpected Output for THIS Guide Image (scores are unknown, use provided clock/period):\n```json\n")
                    dynamic_output = {
                        "home_score": self.dynamic_example_data.get("home_score"), 
                        "away_score": self.dynamic_example_data.get("away_score"), 
                        "clock": self.dynamic_example_data.get("clock"),
                        "period": self.dynamic_example_data.get("period"),
                        "confidence": self.dynamic_example_data.get("confidence", { "home_score": 99, "away_score": 99, "clock": 99, "period": 99 })
                    }
                    content.append(json.dumps(dynamic_output, indent=2))
                    content.append("\n```\n--- End of Detailed Guide ---\n\n")
                    dynamic_example_prepared_successfully = True # Mark as successful
            else:
                logger.warning(f"Could not load dynamic guide image: {self.dynamic_example_image_path}. Skipping dynamic guide.")

        # 3. Final instruction for the ACTUAL image to be analyzed
        if dynamic_example_prepared_successfully: # This refers to the class-level dynamic example
            content.append("Now, using all the guidance above (especially the detailed guide if provided, and any preceding frame context), analyze THIS new scoreboard image:\n")
        elif prev_frame_guide_image and prev_frame_guide_data: # If previous frame context was given
            content.append("Now, using the preceding frame context and the general examples, analyze THIS new scoreboard image:\n")
        else: # No dynamic examples of any kind
            content.append("Now, using the general examples, analyze THIS new scoreboard image:\n")

        if custom_prompt_for_current_image: # New block to add custom prompt
            content.append(f"\nImportant context for the image you are about to analyze:\n{custom_prompt_for_current_image}\n")

        content.append("Input:")
        test_img = self._load_image(image_path)
        if not test_img:
            logger.error(f"Could not load test image: {image_path}")
            return {"error": f"Could not load test image: {image_path}", "success": False}
        content.append(test_img)
        content.append("\nOutput:") # To guide model for its own output structure

        # Make the call to the generative model (unchanged)
        response = self.model.generate_content(content)
        extraction = self._extract_json_from_response(response.text)
        if not extraction:
            extraction = {"raw_response": response.text, "error": "Failed to extract JSON."}
        
        if return_raw_output:
            extraction["raw_output"] = response.text
        
        self._save_to_cache(image_path, extraction)
        return extraction
    
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
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Extract scoreboard information from basketball game images")
    parser.add_argument("--image", type=str, help="Path to a single image file to be analyzed")
    parser.add_argument("--image-dir", type=str, help="Directory containing images to process")
    parser.add_argument("--output", type=str, help="Output JSON file for batch processing")
    parser.add_argument("--examples-path", type=str, default="accumulated_scoreboard_results/accumulated_examples/examples.json",
                      help="Path to examples JSON file")
    parser.add_argument("--examples-dir", type=str, default="accumulated_scoreboard_results/accumulated_examples",
                      help="Directory containing example images")
    parser.add_argument("--api-key", type=str, help="Google API key (defaults to GOOGLE_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers for batch processing")
    parser.add_argument("--confidence-threshold", type=float, default=80.0,
                      help="Minimum confidence score to accept a prediction")
    parser.add_argument("--file-pattern", type=str, default="*.jpg,*.png",
                      help="Comma-separated list of file patterns to process")
    parser.add_argument("--no-cache", action="store_true", help="Disable result caching")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--use-images", action="store_true", default=True,
                      help="Use images in examples (set to False for text-only mode)")

    # New arguments for dynamic example
    parser.add_argument("--dynamic-image", type=str, help="Path to image for the dynamic example (e.g., test_data/frame_0.jpg)")
    parser.add_argument("--dynamic-calibration", type=str, help="Path to calibration file for the dynamic example")
    parser.add_argument("--dynamic-data-json", type=str, help="JSON string with ground truth for dynamic example (e.g., '{\"home_score\":null, \"clock\":\"01:23\"}')")
    parser.add_argument("--dynamic-custom-description", type=str, help="Custom textual description of the dynamic example scoreboard layout.")

    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified for analysis.")
    
    if args.image_dir and not args.output:
        parser.error("--output must be specified when using --image-dir")

    dynamic_example_data_dict = None
    if args.dynamic_data_json:
        try:
            dynamic_example_data_dict = json.loads(args.dynamic_data_json)
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON string for --dynamic-data-json: {e}")
    
    # Check if all parts of dynamic example data (image, data) are provided if one is
    # Calibration and custom description are optional enhancements to the dynamic example if image & data are given
    if (args.dynamic_image and not dynamic_example_data_dict) or \
       (not args.dynamic_image and dynamic_example_data_dict):
        parser.error("Both --dynamic-image and --dynamic-data-json must be provided together if one is specified for a dynamic example.")
    
    # If dynamic_image is provided, then dynamic_calibration can be provided (optional for drawing)
    # and dynamic_custom_description can be provided (optional text)

    recognizer = ScoreboardRecognizer(
        examples_path=args.examples_path,
        examples_dir=args.examples_dir,
        api_key=args.api_key,
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        cache_results=not args.no_cache,
        use_images=args.use_images,
        dynamic_example_image_path=args.dynamic_image,
        dynamic_example_calibration_path=args.dynamic_calibration,
        dynamic_example_data=dynamic_example_data_dict,
        dynamic_example_custom_description=args.dynamic_custom_description
    )
    
    if args.image:
        result = recognizer.extract_scoreboard(args.image, return_raw_output=args.verbose)
        print(json.dumps(result, indent=2))
        return
    
    if args.image_dir:
        # In batch mode, the dynamic example will be included in prompts for ALL images in the batch.
        recognizer.extract_and_save_batch(
            image_dir=args.image_dir,
            output_file=args.output,
            file_pattern=args.file_pattern,
            max_workers=args.max_workers,
            show_progress=True # Assuming True is a good default for CLI
        )
        # The extract_and_save_batch returns (results, success_count, total_count), 
        # but we don't have specific print logic for these return values here yet.
        # logger.info might have already printed summary.
        print(f"Batch processing complete. Results saved to {args.output}")
        return

if __name__ == "__main__":
    main() 