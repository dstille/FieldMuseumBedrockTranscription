"""
Model factory for AWS Bedrock image processors.
This module provides a factory class to create the appropriate processor for a given model.
"""

import boto3
import os
from models.anthropic import AnthropicImageProcessor
from models.meta import MetaImageProcessor
from models.mistral import MistralImageProcessor
from models.cohere import CohereImageProcessor
from models.amazon import AmazonTitanImageProcessor
from models.ai21 import AI21ImageProcessor

class ModelFactory:
    """Factory class for creating model-specific processors."""
    
    @staticmethod
    def get_processor(api_key, prompt_name, prompt_text, model, modelname):
        """
        Get the appropriate processor for the given model.
        
        Args:
            api_key: API key (not used for AWS Bedrock, but kept for compatibility)
            prompt_name: Name of the prompt
            prompt_text: Text of the prompt
            model: Model ID
            modelname: Display name of the model
            
        Returns:
            An instance of the appropriate processor class
        """
        if "anthropic.claude" in model:
            return AnthropicImageProcessor(api_key, prompt_name, prompt_text, model, modelname)
        elif "meta.llama" in model:
            return MetaImageProcessor(api_key, prompt_name, prompt_text, model, modelname)
        elif "mistral." in model:
            return MistralImageProcessor(api_key, prompt_name, prompt_text, model, modelname)
        elif "cohere." in model:
            return CohereImageProcessor(api_key, prompt_name, prompt_text, model, modelname)
        elif "amazon.titan" in model:
            return AmazonTitanImageProcessor(api_key, prompt_name, prompt_text, model, modelname)
        elif "ai21." in model:
            return AI21ImageProcessor(api_key, prompt_name, prompt_text, model, modelname)
        else:
            # Default to Anthropic for unknown models
            print(f"Warning: Unknown model {model}, defaulting to Anthropic processor")
            return AnthropicImageProcessor(api_key, prompt_name, prompt_text, model, modelname)
    
    @staticmethod
    def get_image_capable_models():
        """
        Get a list of model families that support image processing.
        
        Returns:
            A list of model family prefixes that support image processing
        """
        return [
            "anthropic.claude-3",
            "anthropic.claude-3-5",
            "anthropic.claude-3-7",
            "meta.llama4",
            "mistral.pixtral"
        ]
    
    @staticmethod
    def model_supports_images(model_id):
        """
        Check if a model supports image processing.
        
        Args:
            model_id: The model ID to check
            
        Returns:
            True if the model supports image processing, False otherwise
        """
        image_capable_models = ModelFactory.get_image_capable_models()
        
        for model_prefix in image_capable_models:
            if model_prefix in model_id:
                return True
        
        return False
    
    @staticmethod
    def filter_image_capable_models(model_ids):
        """
        Filter a list of model IDs to only include those that support image processing.
        
        Args:
            model_ids: A list of model IDs to filter
            
        Returns:
            A list of model IDs that support image processing
        """
        # First filter by capability
        capable_models = [
            model_id for model_id in model_ids
            if ModelFactory.model_supports_images(model_id)
        ]
        
        # Then filter by availability (check which models are actually accessible)
        available_models = []
        for model_id in capable_models:
            if ModelFactory.is_model_available(model_id):
                available_models.append(model_id)
            else:
                print(f"Warning: Model {model_id} supports images but is not available in your account or region")
        
        return available_models
    
    @staticmethod
    def is_model_available(model_id):
        """
        Check if a model is available in the current AWS account and region.
        
        Args:
            model_id: The model ID to check
            
        Returns:
            True if the model is available, False otherwise
        """
        try:
            # Initialize Bedrock client
            bedrock = boto3.client(
                'bedrock',
                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            # Try to get the model
            response = bedrock.get_foundation_model(modelIdentifier=model_id)
            
            # If we get here, the model exists and is accessible
            return True
            
        except Exception as e:
            # If we get an error, the model might not be available
            print(f"Model availability check for {model_id}: {str(e)}")
            return False
    
    @staticmethod
    def get_available_inference_profiles():
        """
        Get a list of available inference profiles in the current AWS account and region.
        
        Returns:
            A list of inference profile ARNs
        """
        try:
            # Initialize Bedrock client
            bedrock = boto3.client(
                'bedrock',
                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            # List inference profiles
            response = bedrock.list_inference_profiles()
            
            # Extract profile ARNs
            profiles = []
            for profile in response.get('inferenceProfiles', []):
                profiles.append({
                    'name': profile.get('name'),
                    'arn': profile.get('inferenceProfileArn'),
                    'models': profile.get('foundationModelArns', [])
                })
            
            return profiles
            
        except Exception as e:
            print(f"Error getting inference profiles: {str(e)}")
            return []