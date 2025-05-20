# Image-Based Prompt for Scoreboard Recognition

This directory contains the files needed to use the best image-based prompt for scoreboard recognition.

## Contents:
- `examples.json`: Metadata for the examples
- `example_*.jpg`: Example images

## How to use:
1. Install required libraries: `pip install google-generativeai pillow`
2. Set your API key: `export GOOGLE_API_KEY=your_api_key`
3. Use the provided code to create an image-based prompt and process new images:

```python
import os
import json
from PIL import Image
import google.generativeai as genai
import re

def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_json_from_response(text):
    # Look for JSON pattern in the response
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Could not decode JSON: {json_str}")
    
    # Try to find JSON without markers
    try:
        # Try to load the entire response as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def extract_scoreboard(image_path, examples_dir="best_examples"):
    # Configure API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    
    # Load example metadata
    with open(os.path.join(examples_dir, "examples.json"), "r") as f:
        examples = json.load(f)
    
    # Create prompt content
    content = []
    content.append("You are an expert at analyzing basketball scoreboards.")
    content.append("Extract the following information: home_score, away_score, clock, and period.")
    content.append("Here are some examples:\n\n")
    
    # Add each example
    for i, example in enumerate(examples, 1):
        image_id = example["image_id"]
        image_path = os.path.join(examples_dir, f"example_{image_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Example image {image_id} not found, skipping")
            continue
        
        # Add example
        content.append(f"Example #{i}\nInput: ")
        img = load_image(image_path)
        if img:
            content.append(img)
        else:
            continue
            
        # Add output
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
    test_img = load_image(image_path)
    if not test_img:
        raise ValueError(f"Could not load test image: {image_path}")
    content.append(test_img)
    
    # Run the model
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(content)
    
    # Extract JSON from response
    extraction = extract_json_from_response(response.text)
    return extraction or {"raw_response": response.text}

# Example usage
result = extract_scoreboard("path/to/image.jpg")
print(json.dumps(result, indent=2))
```
