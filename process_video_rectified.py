#!/usr/bin/env python3
"""
Processes a video, extracts scoreboard information frame by frame using ScoreboardRecognizer,
and writes the results onto a new output video.
Dynamically generates a scoreboard layout description from the first frame.
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import json
import logging
from pathlib import Path
import time
from typing import Optional, Dict, Any

# Add the parent directory to sys.path to allow importing scoreboard_recognition
import sys
# Assuming scoreboard_recognition.py is in the same directory or accessible in PYTHONPATH
# If it's in the parent directory:
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scoreboard_recognition import ScoreboardRecognizer
    import google.generativeai as genai # For dynamic description
except ImportError:
    print("Error: scoreboard_recognition.py or google.generativeai not found. Make sure they are installed and accessible.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('process_video_rectified')

def draw_text_on_frame(frame, data: dict, font_scale=0.7, thickness=2):
    """Draws extracted scoreboard data on the frame."""
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)  # White
    
    if not data or not isinstance(data, dict):
        text = "Error: No data or invalid data format"
        cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        return

    # Prioritize extracted fields if present
    home_score = data.get("home_score", "N/A")
    away_score = data.get("away_score", "N/A")
    clock = data.get("clock", "N/A")
    period = data.get("period", "N/A")

    texts = [
        f"Home: {home_score}",
        f"Away: {away_score}",
        f"Clock: {clock}",
        f"Period: {period}"
    ]
    
    if "error" in data:
        texts.append(f"Error: {data['error'][:50]}...") # Truncate long errors

    for text in texts:
        cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += 30

def generate_dynamic_scoreboard_description(
    first_frame_image: Image.Image, 
    api_key: str, 
    model_name: str = "gemini-1.5-flash-latest", # Default, can be overridden
    example_image_for_description_task: Optional[Image.Image] = None,
    example_text_for_description_task: Optional[str] = None
) -> Optional[str]:
    """
    Uses Gemini to generate a textual description of the scoreboard layout
    from the first frame of the video.
    Can be guided by a one-shot example.
    """
    logger.info(f"Attempting to dynamically generate scoreboard description using model: {model_name}")
    
    if not api_key:
        logger.error("API key not provided for dynamic description generation.")
        return None
        
    try:
        genai.configure(api_key=api_key)
        desc_model = genai.GenerativeModel(model_name)
        
        prompt_parts = []
        prompt_parts.append("You are an expert at analyzing sports video feeds to understand scoreboard layouts.")
        prompt_parts.append("Your task is to describe the visual layout of the scoreboard in a given image.")
        prompt_parts.append("Focus on describing WHERE the home score, away score, game clock, and period are located, and any distinguishing visual characteristics.")
        prompt_parts.append("This description will be used to help another AI extract these values accurately from subsequent frames of the same game.")
        prompt_parts.append("Do NOT extract the actual values (scores, time, period) in your description.")
        prompt_parts.append("Provide a concise, factual paragraph about the layout.")

        if example_image_for_description_task and example_text_for_description_task:
            logger.info("Using provided one-shot example for description generation.")
            prompt_parts.append("\nHere is an example of how to describe a scoreboard:")
            prompt_parts.append("Example Input Image:")
            prompt_parts.append(example_image_for_description_task)
            prompt_parts.append("Example Output Description:")
            prompt_parts.append(example_text_for_description_task)
            prompt_parts.append("\nNow, based on that example, describe the layout in the following new image.")
        
        prompt_parts.append("\nInput Image to Describe:")
        prompt_parts.append(first_frame_image)
        prompt_parts.append("Output Description:")
        
        response = desc_model.generate_content(prompt_parts)
        
        if response.text:
            description = response.text.strip()
            logger.info(f"Dynamically generated scoreboard description: {description}")
            return description
        else:
            logger.warning("Dynamic scoreboard description generation returned an empty response.")
            return None
            
    except Exception as e:
        logger.error(f"Error during dynamic scoreboard description generation: {e}")
        # You might want to inspect `e` further, e.g., if it's an API error for model not found.
        if "404" in str(e) and "is not found" in str(e): # Basic check for model not found
             logger.error(f"The model {model_name} for description generation might not be valid or available. Please check.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process video to extract scoreboard data and display it.")
    parser.add_argument("--input-video", type=str, default="rectified_output.mp4",
                        help="Path to the input video file.")
    parser.add_argument("--output-video", type=str, required=True,
                        help="Path to save the output video file.")
    
    # Arguments for ScoreboardRecognizer
    parser.add_argument("--examples-path", type=str, default="accumulated_scoreboard_results/accumulated_examples/examples.json",
                        help="Path to examples JSON file for ScoreboardRecognizer.")
    parser.add_argument("--examples-dir", type=str, default="accumulated_scoreboard_results/accumulated_examples",
                        help="Directory containing example images for ScoreboardRecognizer.")
    parser.add_argument("--api-key", type=str, help="Google API key (defaults to GOOGLE_API_KEY environment variable). Required.")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash-latest", 
                        help="Gemini model to use for ScoreboardRecognizer data extraction.")
    parser.add_argument("--description-model", type=str, default="gemini-1.5-flash-latest",
                        help="Gemini model to use for dynamic scoreboard description generation.")
    parser.add_argument("--no-cache", action="store_true", help="Disable result caching in ScoreboardRecognizer.")
    parser.add_argument("--use-images", action="store_false",
                        help="Use images in examples for ScoreboardRecognizer (defaults to True). Pass flag to set to False.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--temporal-context-prompt", type=str, 
                        default="In sports, scores and the clock typically change incrementally between consecutive frames. Large, sudden jumps in scores are rare unless it's a major event like the end of a period or a significant play. Please use the previous frame\'s data as a strong reference.",
                        help="Text prompt explaining temporal consistency to the model.")
    parser.add_argument("--max-processing-time", type=float, default=None,
                        help="Maximum video duration (in seconds) to process (e.g., 20.0 for first 20s). Default is full video.")
    parser.add_argument("--desc-example-image-path", type=str, default=None,
                        help="Path to an example image for the description generation task (e.g., test_data/frame_0.jpg).")
    parser.add_argument("--desc-example-text", type=str, default=None,
                        help="Textual description corresponding to --desc-example-image-path.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('scoreboard_recognition').setLevel(logging.DEBUG)
    
    if not args.api_key:
        args.api_key = os.environ.get("GOOGLE_API_KEY")
        if not args.api_key:
            logger.error("API key not provided via --api-key or GOOGLE_API_KEY environment variable. It is required.")
            return

    if not os.path.exists(args.input_video):
        logger.error(f"Input video not found: {args.input_video}")
        return

    try:
        recognizer = ScoreboardRecognizer(
            examples_path=args.examples_path,
            examples_dir=args.examples_dir,
            api_key=args.api_key, # API key is now passed here
            model_name=args.model,
            cache_results=not args.no_cache,
            use_images=args.use_images
        )
    except ValueError as e:
        logger.error(f"Failed to initialize ScoreboardRecognizer: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during ScoreboardRecognizer initialization: {e}")
        return

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {args.input_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    # Using 'mp4v' for .mp4 output, or 'XVID' for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))

    logger.info(f"Processing video: {args.input_video} ({fps:.2f} FPS, {frame_width}x{frame_height}, {total_frames} frames)")
    logger.info(f"Output video will be saved to: {args.output_video}")

    # Dynamically generate scoreboard description from the first frame
    dynamic_scoreboard_desc: Optional[str] = None
    if total_frames > 0:
        ret_first, first_frame_cv = cap.read()
        if ret_first:
            first_frame_pil = Image.fromarray(cv2.cvtColor(first_frame_cv, cv2.COLOR_BGR2RGB))
            
            # Load example for description task if provided
            example_image_for_desc = None
            if args.desc_example_image_path:
                if os.path.exists(args.desc_example_image_path):
                    try:
                        example_image_for_desc = Image.open(args.desc_example_image_path)
                        logger.info(f"Loaded example image for description task: {args.desc_example_image_path}")
                    except Exception as e:
                        logger.warning(f"Could not load example image {args.desc_example_image_path}: {e}")
                else:
                    logger.warning(f"Example image for description task not found: {args.desc_example_image_path}")
            
            # Check if both example image and text are provided for the description task
            final_example_image_for_desc = None
            final_example_text_for_desc = None
            if example_image_for_desc and args.desc_example_text:
                final_example_image_for_desc = example_image_for_desc
                final_example_text_for_desc = args.desc_example_text
            elif args.desc_example_image_path or args.desc_example_text:
                # Only one part of the example was given
                logger.warning("For description task example, both --desc-example-image-path and --desc-example-text must be provided. Ignoring partial example.")

            dynamic_scoreboard_desc = generate_dynamic_scoreboard_description(
                first_frame_pil, 
                args.api_key,
                model_name=args.description_model,
                example_image_for_description_task=final_example_image_for_desc,
                example_text_for_description_task=final_example_text_for_desc
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            logger.warning("Could not read the first frame for dynamic description generation.")
    else:
        logger.warning("Video has no frames, skipping dynamic description generation.")

    if not dynamic_scoreboard_desc:
        logger.warning("Failed to generate dynamic scoreboard description. Proceeding without specific layout prompt.")
        # You could set a very generic default here if desired
        # dynamic_scoreboard_desc = "A standard basketball scoreboard."

    frame_count = 0
    processed_frame_count = 0
    
    # Calculate how many frames to skip to sample once per second
    # If fps is 0 or invalid, default to processing every 25th frame (arbitrary)
    frames_to_skip_per_sample = int(fps) if fps > 0 else 25 
    logger.info(f"Sampling one frame every {frames_to_skip_per_sample} frames (approx. 1 per second).")

    temp_dir = tempfile.mkdtemp()
    
    last_extraction_time = 0
    min_interval_between_api_calls = 1.0 # seconds, to prevent overwhelming API for very fast processing

    # Variables to store previous frame data for temporal context
    prev_pil_image_for_guide: Optional[Image.Image] = None
    prev_extracted_data_for_guide: Optional[Dict[str, Any]] = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_video_time = frame_count / fps if fps > 0 else 0

            if args.max_processing_time is not None and current_video_time > args.max_processing_time:
                logger.info(f"Reached processing time limit ({args.max_processing_time:.2f}s). Stopping.")
                break

            # Process one frame per second of video time
            if frame_count % frames_to_skip_per_sample == 0:
                logger.info(f"Processing frame {frame_count} (video time: {current_video_time:.2f}s)")
                
                # Ensure we don't call API too frequently if video processing is faster than 1s
                current_time = time.time()
                if current_time - last_extraction_time < min_interval_between_api_calls and processed_frame_count > 0:
                    logger.debug(f"Skipping API call for frame {frame_count} due to minimum interval.")
                    # Still draw previous data if available, or indicate skipping
                    # For simplicity, we'll just write the frame as is or with last data.
                    # If we want to be more precise, we'd store 'last_data'
                    draw_text_on_frame(frame, {"status": "Skipped due to rate limit"}) # Placeholder
                    out.write(frame)
                    frame_count += 1
                    continue
                
                last_extraction_time = current_time

                # Convert frame to PIL Image and save temporarily
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                temp_image_path = os.path.join(temp_dir, f"frame_{processed_frame_count}.png")
                pil_image.save(temp_image_path)

                try:
                    extracted_data = recognizer.extract_scoreboard(
                        temp_image_path,
                        custom_prompt_for_current_image=dynamic_scoreboard_desc, # Use dynamically generated desc
                        prev_frame_guide_image=prev_pil_image_for_guide,        # Pass previous image
                        prev_frame_guide_data=prev_extracted_data_for_guide,    # Pass previous data
                        prev_frame_guide_context_prompt=args.temporal_context_prompt # Pass context prompt
                    )
                    logger.debug(f"Frame {frame_count} - Extracted: {extracted_data}")
                    if not isinstance(extracted_data, dict): # Handle unexpected return types
                        logger.warning(f"Frame {frame_count} - Unexpected data type from recognizer: {type(extracted_data)}. Treating as error.")
                        extracted_data = {"error": "Recognizer returned non-dict data."}
                    
                    # If extraction was successful (or at least not a complete failure to get a dict),
                    # store current frame's image and data for the next iteration.
                    # We store the PIL image directly, not the path.
                    if extracted_data and "error" not in extracted_data: # Basic check for usability
                        prev_pil_image_for_guide = pil_image.copy() # Store a copy
                        prev_extracted_data_for_guide = extracted_data.copy()
                    elif "error" in extracted_data and prev_extracted_data_for_guide: # If current fails, but prev exists
                        # Decide if we want to clear prev_extracted_data_for_guide or keep it.
                        # Keeping it might be better than no guide, but could propagate an old state if many frames fail.
                        # For now, let's clear it if the current frame has an error, to avoid stale guidance after a glitch.
                        logger.debug(f"Clearing previous frame guide due to error in current frame {frame_count}.")
                        prev_pil_image_for_guide = None
                        prev_extracted_data_for_guide = None

                except Exception as e:
                    logger.error(f"Error extracting scoreboard for frame {frame_count} ({temp_image_path}): {e}")
                    extracted_data = {"error": f"Extraction failed: {str(e)}"}
                
                draw_text_on_frame(frame, extracted_data)
                os.remove(temp_image_path) # Clean up temp file
                processed_frame_count += 1
            else:
                # For frames not processed by Gemini, we can choose to draw last known data,
                # or nothing, or a "not processed" message.
                # For now, just write the frame as is without new text.
                # If you want to persist last known data, you'd need to store `extracted_data` from the last successful call.
                pass # Or draw last known data if needed

            out.write(frame)
            frame_count += 1

            if frame_count % (int(fps) * 10) == 0: # Log progress every 10 seconds of video
                 logger.info(f"Progress: Processed up to frame {frame_count}/{total_frames} ({current_video_time:.2f}s)")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # Clean up temp directory
        try:
            for f_name in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f_name))
            os.rmdir(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory {temp_dir}: {e}")

    logger.info(f"Finished processing. Output video saved to {args.output_video}")
    logger.info(f"Total frames in source: {total_frames}. Frames sent to API: {processed_frame_count}")
    if dynamic_scoreboard_desc:
        logger.info(f"Used dynamically generated scoreboard description for processing.")

if __name__ == "__main__":
    main() 