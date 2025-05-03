#!/usr/bin/env python3
"""
Filter AWS Bedrock models to identify those that support image processing.
This script reads available_models.json and creates selected_models.json with only image-capable models.
"""

import json
import os
import argparse
from model_factory import ModelFactory

def load_available_models(file_path="available_models.json"):
    """Load all available models from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return []

def save_selected_models(models, file_path="selected_models.json"):
    """Save selected models to JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(models, f, indent=2)
        print(f"Saved {len(models)} models to {file_path}")
    except Exception as e:
        print(f"Error saving models: {str(e)}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Filter AWS Bedrock models to identify those that support image processing"
    )
    
    parser.add_argument(
        "--input",
        default="available_models.json",
        help="Input JSON file with all available models (default: available_models.json)"
    )
    
    parser.add_argument(
        "--output",
        default="selected_models.json",
        help="Output JSON file for selected models (default: selected_models.json)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all models, not just image-capable ones"
    )
    
    args = parser.parse_args()
    
    # Load all available models
    all_models = load_available_models(args.input)
    
    if not all_models:
        print("No models found in input file.")
        return 1
    
    # Filter models by image capability
    if args.all:
        selected_models = all_models
        print(f"Including all {len(selected_models)} models")
    else:
        selected_models = ModelFactory.filter_image_capable_models(all_models)
        print(f"Found {len(selected_models)} image-capable models out of {len(all_models)} total models")
    
    # Save selected models
    save_selected_models(selected_models, args.output)
    
    # Print the selected models
    print("\nSelected models:")
    for model in selected_models:
        print(f"- {model}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())