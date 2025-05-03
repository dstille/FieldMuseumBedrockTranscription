#!/usr/bin/env python3
"""
Model Capabilities Utility for AWS Bedrock
This script helps identify which models support image processing and other capabilities.
"""

import json
import os
import boto3
import argparse
from tabulate import tabulate

# Define model capabilities
MODEL_CAPABILITIES = {
    # Anthropic models
    "anthropic.claude-3-opus": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 200000,
        "description": "Claude 3 Opus - Most powerful Claude model"
    },
    "anthropic.claude-3-sonnet": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 200000,
        "description": "Claude 3 Sonnet - Balanced Claude model"
    },
    "anthropic.claude-3-haiku": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 200000,
        "description": "Claude 3 Haiku - Fastest Claude model"
    },
    "anthropic.claude-3-5-sonnet": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 200000,
        "description": "Claude 3.5 Sonnet - Latest Claude model"
    },
    "anthropic.claude-3-7-sonnet": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 200000,
        "description": "Claude 3.7 Sonnet - Latest Claude model"
    },
    "anthropic.claude-2": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 100000,
        "description": "Claude 2 - Text-only model"
    },
    
    # Meta models
    "meta.llama3-8b-instruct": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Llama 3 8B - Smaller Llama model"
    },
    "meta.llama3-70b-instruct": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Llama 3 70B - Larger Llama model"
    },
    "meta.llama4-scout-17b-instruct": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 128000,
        "description": "Llama 4 Scout - Vision-capable model"
    },
    "meta.llama4-maverick-17b-instruct": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 128000,
        "description": "Llama 4 Maverick - Vision-capable model"
    },
    
    # Mistral models
    "mistral.mistral-7b-instruct": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Mistral 7B - Smaller Mistral model"
    },
    "mistral.mixtral-8x7b-instruct": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Mixtral 8x7B - Mixture of experts model"
    },
    "mistral.mistral-large": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Mistral Large - Larger Mistral model"
    },
    "mistral.pixtral-large": {
        "image_processing": True,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Pixtral Large - Vision-capable Mistral model"
    },
    
    # Cohere models
    "cohere.command-text": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 4096,
        "description": "Cohere Command - Text generation model"
    },
    "cohere.command-r": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 4096,
        "description": "Cohere Command-R - Reasoning-focused model"
    },
    
    # AI21 models
    "ai21.jamba-instruct": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Jamba Instruct - AI21's instruction model"
    },
    
    # Amazon models
    "amazon.titan-text-express": {
        "image_processing": False,
        "json_output": True,
        "max_tokens": 8192,
        "description": "Titan Text Express - Amazon's text model"
    },
    "amazon.titan-image-generator": {
        "image_processing": True,
        "json_output": False,
        "max_tokens": 0,
        "description": "Titan Image Generator - Creates images, not for analysis"
    }
}

def get_model_family(model_id):
    """Extract the model family from a model ID"""
    # Remove version and other suffixes
    parts = model_id.split('-')
    if len(parts) <= 1:
        parts = model_id.split('.')
        if len(parts) > 1:
            return parts[0] + '.' + parts[1]
        return model_id
    
    # Handle special cases
    if "claude" in model_id:
        if "claude-3-5" in model_id:
            return "anthropic.claude-3-5-sonnet"
        elif "claude-3-7" in model_id:
            return "anthropic.claude-3-7-sonnet"
        elif "claude-3-opus" in model_id:
            return "anthropic.claude-3-opus"
        elif "claude-3-sonnet" in model_id:
            return "anthropic.claude-3-sonnet"
        elif "claude-3-haiku" in model_id:
            return "anthropic.claude-3-haiku"
        elif "claude-2" in model_id:
            return "anthropic.claude-2"
    
    # Handle Llama models
    if "llama3" in model_id:
        if "70b" in model_id:
            return "meta.llama3-70b-instruct"
        return "meta.llama3-8b-instruct"
    
    if "llama4" in model_id:
        if "scout" in model_id:
            return "meta.llama4-scout-17b-instruct"
        return "meta.llama4-maverick-17b-instruct"
    
    # Handle Mistral models
    if "mistral" in model_id:
        if "large" in model_id:
            return "mistral.mistral-large"
        return "mistral.mistral-7b-instruct"
    
    if "mixtral" in model_id:
        return "mistral.mixtral-8x7b-instruct"
    
    if "pixtral" in model_id:
        return "mistral.pixtral-large"
    
    # Default to first two parts
    return '.'.join(parts[:2])

def get_model_capabilities(model_id):
    """Get capabilities for a specific model"""
    model_family = get_model_family(model_id)
    
    # Try to find an exact match
    if model_family in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model_family]
    
    # Try to find a partial match
    for family, capabilities in MODEL_CAPABILITIES.items():
        if family in model_id:
            return capabilities
    
    # Default capabilities
    return {
        "image_processing": False,
        "json_output": False,
        "max_tokens": 4096,
        "description": "Unknown model capabilities"
    }

def list_available_models(region=None, profile=None):
    """List all available models in the AWS account"""
    session = boto3.Session(profile_name=profile, region_name=region or os.getenv('AWS_REGION', 'us-east-1'))
    bedrock = session.client('bedrock')
    
    try:
        response = bedrock.list_foundation_models()
        return response.get('modelSummaries', [])
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def filter_image_capable_models(models):
    """Filter models that support image processing"""
    image_models = []
    
    for model in models:
        model_id = model.get('modelId')
        capabilities = get_model_capabilities(model_id)
        
        if capabilities.get('image_processing', False):
            image_models.append({
                'modelId': model_id,
                'description': capabilities.get('description', ''),
                'max_tokens': capabilities.get('max_tokens', 0),
                'json_output': capabilities.get('json_output', False)
            })
    
    return image_models

def save_models_to_json(models, output_file):
    """Save model IDs to a JSON file"""
    model_ids = [model.get('modelId') for model in models]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(model_ids, f, indent=4)
    
    print(f"Saved {len(model_ids)} models to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AWS Bedrock Model Capabilities Utility")
    
    parser.add_argument(
        "--region",
        help="AWS region (default: from environment or us-east-1)"
    )
    
    parser.add_argument(
        "--profile",
        help="AWS profile name"
    )
    
    parser.add_argument(
        "--filter",
        choices=["all", "image", "json"],
        default="all",
        help="Filter models by capability (default: all)"
    )
    
    parser.add_argument(
        "--output",
        help="Output JSON file for model IDs"
    )
    
    args = parser.parse_args()
    
    # List available models
    models = list_available_models(args.region, args.profile)
    
    if not models:
        print("No models found or error accessing AWS Bedrock")
        return 1
    
    # Apply filters
    filtered_models = []
    if args.filter == "image":
        filtered_models = filter_image_capable_models(models)
    elif args.filter == "json":
        filtered_models = [
            {
                'modelId': model.get('modelId'),
                'description': get_model_capabilities(model.get('modelId')).get('description', ''),
                'json_output': get_model_capabilities(model.get('modelId')).get('json_output', False)
            }
            for model in models
            if get_model_capabilities(model.get('modelId')).get('json_output', False)
        ]
    else:
        filtered_models = [
            {
                'modelId': model.get('modelId'),
                'description': get_model_capabilities(model.get('modelId')).get('description', ''),
                'image_processing': get_model_capabilities(model.get('modelId')).get('image_processing', False),
                'json_output': get_model_capabilities(model.get('modelId')).get('json_output', False)
            }
            for model in models
        ]
    
    # Display results
    if args.filter == "image":
        headers = ["Model ID", "Description", "Max Tokens", "JSON Output"]
        table_data = [
            [model['modelId'], model['description'], model['max_tokens'], "Yes" if model['json_output'] else "No"]
            for model in filtered_models
        ]
    elif args.filter == "json":
        headers = ["Model ID", "Description", "JSON Output"]
        table_data = [
            [model['modelId'], model['description'], "Yes" if model['json_output'] else "No"]
            for model in filtered_models
        ]
    else:
        headers = ["Model ID", "Description", "Image Processing", "JSON Output"]
        table_data = [
            [model['modelId'], model['description'], 
             "Yes" if model['image_processing'] else "No",
             "Yes" if model['json_output'] else "No"]
            for model in filtered_models
        ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"Total models: {len(filtered_models)}")
    
    # Save to JSON if requested
    if args.output:
        save_models_to_json(filtered_models, args.output)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())