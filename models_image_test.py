#!/usr/bin/env python3
"""
Model Image Test Script for AWS Bedrock.
This script tests a single image with a prompt across all available models in your AWS account.
"""

import os
import json
import base64
import time
import datetime
import boto3
import re
from pathlib import Path
from bedrock_interface import BedrockImageProcessor

# Configuration - modify these values as needed
IMAGE_PATH = "test_images/test_image.jpg"  # Path to the test image
PROMPT_FILE = "prompts/1.5Stripped.txt"  # Path to the prompt file
OUTPUT_DIR = "test_results"  # Directory to save test results
SAVE_INDIVIDUAL_RESULTS = True  # Whether to save individual model results
TEST_ALL_MODELS = False  # Whether to test all models or only those marked as image-capable
ACCEPT_NON_JSON = True  # Whether to accept non-JSON responses
INCLUDE_NEW_MODELS = True  # Whether to include newly available models

def ensure_directory_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_prompt(prompt_path):
    """Load prompt text from a file."""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading prompt: {str(e)}")
        return None

def image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

def load_available_models():
    """Load available models from selected_models.json."""
    try:
        with open("selected_models.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return []

def get_all_bedrock_models():
    """Get all available models from AWS Bedrock."""
    try:
        # Initialize Bedrock client
        bedrock = boto3.client(
            'bedrock',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # List foundation models
        response = bedrock.list_foundation_models()
        
        # Extract model IDs
        model_ids = []
        for model in response.get('modelSummaries', []):
            model_id = model.get('modelId')
            if model_id:
                model_ids.append(model_id)
        
        return model_ids
    except Exception as e:
        print(f"Error getting Bedrock models: {str(e)}")
        return []

def save_test_results(results, filename):
    """Save test results to a JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def contains_error_message(content):
    """Check if the content contains error messages."""
    if not content:
        return True, "Empty response"
    
    # Check for common error patterns
    error_patterns = [
        r"error",
        r"exception",
        r"fail",
        r"unable to process",
        r"cannot process",
        r"not support",
        r"invalid",
        r"malformed",
        r"Model .* does not support image processing"
    ]
    
    # Convert to string if it's not already
    if not isinstance(content, str):
        content = str(content)
    
    content_lower = content.lower()
    
    for pattern in error_patterns:
        if re.search(pattern, content_lower):
            return True, f"Found error pattern: {pattern}"
    
    # Check if the content is just a raw text wrapper with an error message
    if isinstance(content, str):
        try:
            content_json = json.loads(content)
            if "error" in content_json:
                return True, content_json["error"]
            if "raw_text" in content_json and "error" in content_json:
                return True, content_json["error"]
        except:
            pass
    
    return False, None

def test_models():
    """Test all models with the specified image and prompt."""
    # Ensure output directory exists
    ensure_directory_exists(OUTPUT_DIR)
    
    # Load prompt
    prompt_text = load_prompt(PROMPT_FILE)
    if not prompt_text:
        print("Failed to load prompt. Exiting.")
        return
    
    # Load image
    base64_image = image_to_base64(IMAGE_PATH)
    if not base64_image:
        print("Failed to load image. Exiting.")
        return
    
    # Get image name for reference
    image_name = Path(IMAGE_PATH).stem
    
    # Get models to test
    if TEST_ALL_MODELS:
        # Get all models from AWS Bedrock
        models_to_test = get_all_bedrock_models()
        if not models_to_test:
            # Fall back to selected_models.json if API call fails
            models_to_test = load_available_models()
    else:
        # Load models from selected_models.json
        models_to_test = load_available_models()
    
    # If we want to include new models, get all available models and add any that aren't in our list
    if INCLUDE_NEW_MODELS and not TEST_ALL_MODELS:
        all_models = get_all_bedrock_models()
        for model in all_models:
            if model not in models_to_test:
                # Check if it's likely to be an image-capable model
                if any(family in model for family in ["claude-3", "claude-3-5", "claude-3-7", "llama4", "pixtral", "nova"]):
                    print(f"Adding newly available model: {model}")
                    models_to_test.append(model)
    
    if not models_to_test:
        print("No models found to test. Exiting.")
        return
    
    print(f"Testing {len(models_to_test)} models...")
    
    # Prepare results container
    test_results = {
        "test_id": f"image_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.datetime.now().isoformat(),
        "image": IMAGE_PATH,
        "prompt_file": PROMPT_FILE,
        "models_tested": len(models_to_test),
        "results": []
    }
    
    # Test each model
    for i, model_id in enumerate(models_to_test):
        print(f"\n[{i+1}/{len(models_to_test)}] Testing model: {model_id}")
        
        # Extract model name for display
        model_name_parts = model_id.split('.')
        if len(model_name_parts) > 1:
            model_name = model_name_parts[1].split('-')[0]  # Get the base name (claude, llama, etc.)
        else:
            model_name = model_id
        
        try:
            # Create processor for this model
            processor = BedrockImageProcessor(
                api_key="",  # Empty as we're using AWS credentials from environment
                prompt_name=Path(PROMPT_FILE).stem,
                prompt_text=prompt_text,
                model=model_id,
                modelname=model_name,
                accept_non_json=ACCEPT_NON_JSON
            )
            
            # Process the image
            start_time = time.time()
            content, processing_data = processor.process_image(base64_image, image_name, 0)
            end_time = time.time()
            
            # Calculate processing time
            processing_time = end_time - start_time
            
            # Check if the content contains error messages
            has_error, error_message = contains_error_message(content)
            
            # Parse content as JSON if it's a string
            if isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    parsed_content = {"raw_text": content}
            else:
                parsed_content = content
            
            # Create result entry
            result = {
                "model_id": model_id,
                "model_name": model_name,
                "status": "error" if has_error else "success",
                "processing_time_seconds": processing_time,
                "tokens": {
                    "input": processing_data.get("input tokens", 0) if processing_data else 0,
                    "output": processing_data.get("output tokens", 0) if processing_data else 0,
                    "total": (processing_data.get("input tokens", 0) + processing_data.get("output tokens", 0)) if processing_data else 0
                },
                "costs": {
                    "input": processing_data.get("input cost $", 0) if processing_data else 0,
                    "output": processing_data.get("output cost $", 0) if processing_data else 0,
                    "total": (processing_data.get("input cost $", 0) + processing_data.get("output cost $", 0)) if processing_data else 0
                }
            }
            
            # Add error message if there is one
            if has_error:
                result["error"] = error_message
            
            # Save individual result if enabled
            if SAVE_INDIVIDUAL_RESULTS:
                # Create a safe filename by replacing problematic characters
                safe_model_name = model_name.replace(".", "_").replace(":", "_")
                model_result_file = f"{OUTPUT_DIR}/{safe_model_name}_result.json"
                with open(model_result_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": model_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "result": parsed_content,
                        "metrics": result
                    }, f, indent=2, ensure_ascii=False)
            
            # Add to overall results
            test_results["results"].append(result)
            
            # Print summary
            if has_error:
                print(f"  ✗ Error with {model_name}: {error_message}")
            else:
                print(f"  ✓ Success: {model_name}")
                print(f"    Time: {processing_time:.2f} seconds")
                print(f"    Tokens: {result['tokens']['total']:,}")
                print(f"    Cost: ${result['costs']['total']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error with {model_id}: {str(e)}")
            
            # Add error result
            test_results["results"].append({
                "model_id": model_id,
                "model_name": model_name,
                "status": "error",
                "error": str(e)
            })
    
    # Calculate summary statistics
    successful_tests = [r for r in test_results["results"] if r["status"] == "success"]
    if successful_tests:
        test_results["summary"] = {
            "successful_models": len(successful_tests),
            "failed_models": len(test_results["results"]) - len(successful_tests),
            "fastest_model": min(successful_tests, key=lambda x: x["processing_time_seconds"])["model_id"] if successful_tests else None,
            "slowest_model": max(successful_tests, key=lambda x: x["processing_time_seconds"])["model_id"] if successful_tests else None,
            "cheapest_model": min(successful_tests, key=lambda x: x["costs"]["total"])["model_id"] if successful_tests else None,
            "most_expensive_model": max(successful_tests, key=lambda x: x["costs"]["total"])["model_id"] if successful_tests else None,
            "average_time": sum(r["processing_time_seconds"] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            "average_cost": sum(r["costs"]["total"] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            "total_cost": sum(r["costs"]["total"] for r in successful_tests) if successful_tests else 0
        }
    
    # Save overall results
    results_file = f"{OUTPUT_DIR}/model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_test_results(test_results, results_file)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Models tested: {len(models_to_test)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(test_results['results']) - len(successful_tests)}")
    
    if successful_tests:
        summary = test_results["summary"]
        print(f"\nFastest model: {summary['fastest_model']}")
        print(f"Slowest model: {summary['slowest_model']}")
        print(f"Cheapest model: {summary['cheapest_model']}")
        print(f"Most expensive model: {summary['most_expensive_model']}")
        print(f"Average time: {summary['average_time']:.2f} seconds")
        print(f"Average cost: ${summary['average_cost']:.4f}")
        print(f"Total cost: ${summary['total_cost']:.4f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Update selected_models.json with only the successful models
    if successful_tests:
        successful_model_ids = [r["model_id"] for r in successful_tests]
        print(f"\nUpdating selected_models.json with {len(successful_model_ids)} successful models")
        with open("selected_models.json", "w") as f:
            json.dump(successful_model_ids, f, indent=2)

if __name__ == "__main__":
    test_models()