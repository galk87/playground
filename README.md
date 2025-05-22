# Basketball Scoreboard Recognition System

This is a production-ready system for extracting scoreboards from basketball game images. It uses image-based few-shot prompting with Google's Gemini API to accurately extract:

- Home team score
- Away team score
- Game clock
- Period number

## Features

- **High Accuracy**: Achieves >77% global accuracy (all fields correct) and >95% accuracy for individual scores
- **Few-Shot Learning**: Uses 23 optimized examples accumulated across iterative similarity-based training
- **Confidence Scores**: Provides confidence scores for each prediction
- **Parallel Processing**: Supports batch processing with configurable parallelism
- **Result Caching**: Automatically caches results to avoid redundant API calls
- **Comprehensive Logging**: Detailed logging for production environments
- **Text-Only Mode**: Option to use text descriptions instead of images for examples, reducing API costs

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install google-generativeai pillow tqdm
```

3. Set up a Google API key for the Gemini API:
   - Obtain a key from [Google AI Studio](https://makersuite.google.com/)
   - Set the key as an environment variable:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Command-line Interface

#### Process a single image:

```bash
python scoreboard_recognition.py --image path/to/image.jpg
```

#### Process all images in a directory:

```bash
python scoreboard_recognition.py --image-dir path/to/images/ --output results.json
```

#### Advanced options:

```bash
python scoreboard_recognition.py --image-dir path/to/images/ \
  --api-key your_api_key \
  --model gemini-2.0-flash \
  --max-workers 4 \
  --confidence-threshold 75.0 \
  --file-pattern "*.jpg,*.png" \
  --verbose
```

#### Text-only mode (no images in examples):

```bash
python scoreboard_recognition.py --image path/to/image.jpg --use-images False
```

### Using as a Python Library

```python
from scoreboard_recognition import ScoreboardRecognizer

# Initialize the recognizer with image-based examples
recognizer = ScoreboardRecognizer(
    examples_path="continued_iterations_results/accumulated_examples/examples.json",
    examples_dir="continued_iterations_results/accumulated_examples",
    api_key="your_api_key_here"  # Optional if set in environment
)

# Initialize with text-only examples
text_only_recognizer = ScoreboardRecognizer(
    examples_path="continued_iterations_results/accumulated_examples/examples.json",
    examples_dir="continued_iterations_results/accumulated_examples",
    use_images=False  # Enable text-only mode
)

# Process a single image
result = recognizer.extract_scoreboard("path/to/image.jpg")
print(f"Home score: {result['home_score']}")
print(f"Away score: {result['away_score']}")
print(f"Clock: {result['clock']}")
print(f"Period: {result['period']}")

# Process multiple images in parallel
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = recognizer.batch_extract(image_paths, max_workers=3)
```

### Processing Videos with Dynamic Scoreboard Description

A new script `process_video_rectified.py` is available to process video files. It samples frames, extracts scoreboard information, and creates an output video with the data overlaid. A key feature is its ability to **dynamically generate a description of the scoreboard layout** from the first frame of the video, making it adaptable to different scoreboard designs without manual prompting for each video.

It also uses the previously recognized frame's data as a strong hint for the current frame, improving temporal consistency.

#### Basic Video Processing:

```bash
python process_video_rectified.py --input-video path/to/your_video.mp4 --output-video path/to/output_video.mp4 --api-key YOUR_API_KEY
```

#### Advanced Video Processing Options:

This script utilizes the `ScoreboardRecognizer` internally and shares some of its arguments (like `--model`, `--examples-path`, etc.). Key arguments specific to or important for `process_video_rectified.py` include:

- `--input-video <path>`: Path to the input video file (default: `rectified_output.mp4`).
- `--output-video <path>`: Path to save the output video with overlaid results (required).
- `--api-key <key>`: Your Google API key (can also be set via `GOOGLE_API_KEY` environment variable).
- `--model <model_name>`: Gemini model for scoreboard data extraction (default: `gemini-1.5-flash-latest`).
- `--description-model <model_name>`: Gemini model for dynamic scoreboard description generation (default: `gemini-1.5-flash-latest`).
- `--max-processing-time <seconds>`: Process only the first N seconds of the video (e.g., `20.0`). Default is to process the full video.
- `--temporal-context-prompt "Your prompt text"`: Custom prompt to explain temporal consistency to the model when using the previous frame as a guide. A default prompt is provided.
- `--desc-example-image-path <path>`: Path to an example image (e.g., `test_data/frame_0.jpg`) to guide the dynamic scoreboard description generation task. 
- `--desc-example-text "Textual description"`: The known textual description corresponding to the `--desc-example-image-path`. Both this and the image path must be provided if one is used.

Example with dynamic description guidance and processing the first 30 seconds:

```bash
python process_video_rectified.py \
    --input-video example_game.mp4 \
    --output-video processed_game_30s.mp4 \
    --api-key YOUR_API_KEY \
    --model gemini-1.5-flash-latest \
    --description-model gemini-1.5-flash-latest \
    --max-processing-time 30.0 \
    --desc-example-image-path "test_data/frame_0.jpg" \
    --desc-example-text "Home score top-left, away top-right, clock bottom-center."
```

### Demo Script

For a simple demonstration, use the demo script:

```bash
python demo_scoreboard.py --image path/to/test_image.jpg
```

## Model Performance

The system uses an iteratively improved prompt with examples accumulated across all iterations, resulting in 23 diverse examples. During training, it achieved:

- **Global accuracy**: 77.36% (all fields correct)
- **Home score accuracy**: 98.11%
- **Away score accuracy**: 100.00%
- **Clock accuracy**: 79.25%
- **Period accuracy**: 100.00%

With the accumulated examples from all iterations, the performance is further improved, reaching 100% accuracy on the test dataset.

## How It Works

1. **Image-Based Few-Shot Learning**: The system uses 23 carefully selected example images with their correct labels to teach the model how to extract scoreboard information.

2. **Text-Only Mode**: When enabled, the system uses text descriptions of the examples instead of actual images, which can reduce API costs while maintaining good accuracy.

3. **Iterative Similarity Selection with Accumulated Examples**: The examples were selected through an iterative process that:
   - Tests the model on validation images
   - Identifies mistakes
   - Finds training images similar to the mistakes
   - Adds them to the accumulated set of examples
   - Repeats until performance plateaus
   - Saves all unique examples across all iterations

4. **Confidence Scoring**: The model provides confidence scores for each extracted field, allowing applications to handle low-confidence predictions appropriately.

## Customization

### Using Different Examples

You can create your own set of examples:

```bash
python scoreboard_recognition.py --examples-path my_examples.json --examples-dir my_examples_dir
```

### Accumulating New Examples

You can run the example accumulation script to find and add more examples to your set:

```bash
python accumulate_scoreboard_examples.py --val-dir path/to/validation --train-dir path/to/training --output-dir accumulated_results
```

### Model Selection

The system defaults to `gemini-2.0-flash` for a good balance of speed and accuracy. For higher accuracy, you can use `gemini-2.0-pro`:

```bash
python scoreboard_recognition.py --model gemini-2.0-pro
```

## Limitations

- The system is optimized for basketball scoreboards in the style present in the training data
- Performance may vary on significantly different scoreboard designs
- API calls to Gemini have associated costs and rate limits
- Text-only mode may have slightly lower accuracy than image-based mode but offers cost savings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This system uses Google's Gemini API for vision-language understanding
- It was developed through an iterative similarity-based optimization approach with accumulated examples