import boto3
import base64
import json
import time
import os
from typing import Dict, Any, Tuple, Optional
from botocore.exceptions import ClientError
from llm_interface import ImageProcessor
from utilities.base64_filter import filter_base64, filter_base64_from_dict
from dotenv import load_dotenv

class BedrockImageProcessor(ImageProcessor):
    def __init__(self, api_key, prompt_name, prompt_text, model, modelname, output_name):
        super().__init__(api_key, prompt_name, prompt_text, model, modelname, output_name)
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.bedrock_mgmt = boto3.client("bedrock")
        self.model_info = self.load_model_info()
        self.account_id = self._get_account_id()
        self.set_token_costs_per_mil()
    
    def _get_account_id(self) -> str:
        """Get the AWS account ID."""
        try:
            sts_client = boto3.client('sts')
            return sts_client.get_caller_identity()["Account"]
        except Exception as e:
            print(f"Error getting AWS account ID: {str(e)}")
            return ""
    
    def load_model_info(self) -> Dict[str, Any]:
        """Load model information from vision_model_info.json."""
        try:
            with open("model_info/vision_model_info.json", "r") as f:
                models = json.load(f)
                for model in models:
                    if model.get("modelId") == self.model:
                        return model
                return None
        except Exception as e:
            print(f"Error loading model info: {str(e)}")
            return None
    
    def format_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Format the prompt based on the model provider."""
        # Default implementation for models that don't have a specific formatter
        # Add a request for JSON output to the prompt text if not already present
        prompt_text = self.prompt_text
        if "json" not in prompt_text.lower():
            prompt_text += "\n\nPlease provide the transcription as a JSON object with a 'transcription' field. Do not include any newlines, tabs or returns within individual fields."
        
        # Generic format that works with most models
        return {
            "inputText": prompt_text,
            "inputImage": base64_image,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "temperature": 0.0,
                "topP": 0.9
            }
        }
    
    def extract_text(self, response_body: Dict[str, Any]) -> str:
        """Extract text from response."""
        # Default implementation for models that don't have a specific extractor
        # Try common response formats
        try:
            # Check for common response structures
            if "content" in response_body and isinstance(response_body["content"], list):
                # Claude-like format
                for item in response_body["content"]:
                    if isinstance(item, dict) and "text" in item:
                        return item.get("text", "")
            elif "output" in response_body:
                # Nova-like format
                output = response_body["output"]
                if isinstance(output, dict) and "message" in output:
                    message = output["message"]
                    if isinstance(message, dict) and "content" in message:
                        for item in message["content"]:
                            if isinstance(item, dict) and "text" in item:
                                return item.get("text", "")
            elif "results" in response_body and isinstance(response_body["results"], list):
                # Amazon-like format
                return response_body["results"][0].get("outputText", "")
            elif "generation" in response_body:
                # Meta-like format
                return response_body.get("generation", "")
            elif "text" in response_body:
                # Simple format
                return response_body.get("text", "")
            # If we can't find a known structure, convert the whole response to a string
            return json.dumps(response_body)
        except Exception as e:
            print(f"Error extracting text from response: {str(e)}")
            return f"Error extracting text: {str(e)}"
    
    def needs_inference_profile(self) -> bool:
        """Check if the model uses an inference profile."""
        if not self.model_info:
            return False
        inference_types = self.model_info.get("inferenceTypes", [])
        return "INFERENCE_PROFILE" in inference_types
    
    def get_model_id(self) -> str:
        """Get the inference profile ID for the model."""
        # For models that require inference profiles, construct the ARN
        if self.needs_inference_profile():
            # Extract region from the client configuration
            region = self.bedrock_client.meta.region_name
            region_prefix = region.split('-')[0]  # e.g., "us" from "us-east-1"
            # Get provider from model ID
            provider = self.model.split('.')[0] if '.' in self.model else ""
            # Construct the inference profile ARN with region prefix
            # Format: arn:aws:bedrock:{region}:{account_id}:inference-profile/{region_prefix}.{model_id}
            return f"arn:aws:bedrock:{region}:{self.account_id}:inference-profile/{region_prefix}.{self.model}"
        return self.model
    
    def process_image(self, base64_image: str, image_name: str, image_index: int) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        request_body = self.format_prompt(base64_image)
        provider = self.model.split(".")[0] if "." in self.model else ""
        # TODO: Process differently based on provider, especially meta
        try:
            text, processing_data, raw_response = self._process_with_bedrock(request_body, base64_image, image_name, start_time)
            return text, processing_data, raw_response
        except Exception as e:
            error_message = f"Error processing image: {str(e)}"
            print(error_message)
            return error_message, {"error": error_message}, raw_response
    
    def _process_with_bedrock(self, request_body: Dict[str, Any], base64_image: str, 
                             image_name: str, start_time: float) -> Tuple[str, Dict[str, Any]]:
        model_id = self.get_model_id()
        raw_response = None
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response.get("body").read())
            raw_response = self.save_raw_response(response_body, image_name)
            text = self.extract_text(response_body)
            self.update_usage(response_body)
            # Calculate processing time
            time_elapsed = (time.time() - start_time) / 60  # in minutes
            processing_data = self.get_transcript_processing_data(time_elapsed)
            self.num_processed += 1
            return text, processing_data, raw_response
        except Exception as e:
            error_message = f"Error invoking model {model_id}: {str(e)}"
            print(error_message)
            # Add more context to the error message
            if "AccessDeniedException" in str(e):
                error_message += "\nAccess denied: You may not have permissions to use this model or inference profile."
            elif "ValidationException" in str(e) and "inference profile" in str(e).lower():
                error_message += "\nInference profile error: The inference profile may not be set up correctly."
            elif "ResourceNotFoundException" in str(e):
                error_message += "\nResource not found: The model or inference profile may not exist."
            return error_message, {"error": error_message}, raw_response
    
    def update_usage(self, response_data: Dict[str, Any]):
        """Update token usage from response data."""
        # Default implementation for models that don't have a specific usage extractor
        try:
            # Check for common usage structures
            if "usage" in response_data:
                usage = response_data["usage"]
                # Try different key formats
                if "input_tokens" in usage:
                    self.input_tokens = usage.get("input_tokens", 0)
                elif "inputTokens" in usage:
                    self.input_tokens = usage.get("inputTokens", 0)
                elif "prompt_tokens" in usage:
                    self.input_tokens = usage.get("prompt_tokens", 0)
                elif "inputTokenCount" in usage:
                    self.input_tokens = usage.get("inputTokenCount", 0)
                
                if "output_tokens" in usage:
                    self.output_tokens = usage.get("output_tokens", 0)
                elif "outputTokens" in usage:
                    self.output_tokens = usage.get("outputTokens", 0)
                elif "completion_tokens" in usage:
                    self.output_tokens = usage.get("completion_tokens", 0)
                elif "outputTokenCount" in usage:
                    self.output_tokens = usage.get("outputTokenCount", 0)
        except Exception as e:
            print(f"Error updating usage: {str(e)}")

####### Testing Module  #######
import random
RANDOM_ERROR_THRESHOLD = 0.25
SAMPLE_DIRECTORY  = "testing/raw_llm_responses_for_testing"
SAMPLE_FILE = "httpsfm-digital-assetsfieldmuseumorg2298823C0399179Fjpg-2025-05-31-1632-09-raw.json"

class BedrockImageProcessorTesting(ImageProcessor):
    def __init__(self, api_key, prompt_name, prompt_text, model, modelname, output_name):
        super().__init__(api_key, prompt_name, prompt_text, model, modelname, output_name)
        load_dotenv()
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.bedrock_mgmt = boto3.client("bedrock")
        self.model_info = None
        self.account_id = self._get_account_id()
        self.pricing_data = self.load_pricing_data()
        self.set_token_costs_per_mil()
        self.include_random_error = os.getenv("INCLUDE_RANDOM_ERROR", "False").lower() == "true"
    
    def _get_account_id(self) -> str:
        """Get the AWS account ID."""
        try:
            sts_client = boto3.client('sts')
            return sts_client.get_caller_identity()["Account"]
        except Exception as e:
            print(f"Error getting AWS account ID: {str(e)}")
            return ""
    
    def load_model_info(self) -> Dict[str, Any]:
        """Load model information from vision_model_info.json."""
        try:
            with open("model_info/vision_model_info.json", "r") as f:
                models = json.load(f)
                for model in models:
                    if model.get("modelId") == self.model:
                        return model
                return None
        except Exception as e:
            print(f"Error loading model info: {str(e)}")
            return None

    def format_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Format the prompt based on the model provider."""
        # Default implementation for models that don't have a specific formatter
        # Add a request for JSON output to the prompt text if not already present
        prompt_text = self.prompt_text
        if "json" not in prompt_text.lower():
            prompt_text += "\n\nPlease provide the transcription as a JSON object with a 'transcription' field."
        
        # Generic format that works with most models
        return {
            "inputText": prompt_text,
            "inputImage": base64_image,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "temperature": 0.0,
                "topP": 0.9
            }
        }
    
    def extract_text(self, response_body: Dict[str, Any]) -> str:
        """Extract text from response."""
        # Default implementation for models that don't have a specific extractor
        # Try common response formats
        try:
            # Check for common response structures
            if "content" in response_body and isinstance(response_body["content"], list):
                # Claude-like format
                for item in response_body["content"]:
                    if isinstance(item, dict) and "text" in item:
                        return item.get("text", "")
            elif "output" in response_body:
                # Nova-like format
                output = response_body["output"]
                if isinstance(output, dict) and "message" in output:
                    message = output["message"]
                    if isinstance(message, dict) and "content" in message:
                        for item in message["content"]:
                            if isinstance(item, dict) and "text" in item:
                                return item.get("text", "")
            elif "results" in response_body and isinstance(response_body["results"], list):
                # Amazon-like format
                return response_body["results"][0].get("outputText", "")
            elif "generation" in response_body:
                # Meta-like format
                return response_body.get("generation", "")
            elif "transcription" in response_body:
                return response_body.get("transcription", "")
            elif "text" in response_body:
                # Simple format
                return response_body.get("text", "")
            # If we can't find a known structure, convert the whole response to a string
            return json.dumps(response_body)
        except Exception as e:
            print(f"Error extracting text from response: {str(e)}")
            return f"Error extracting text: {str(e)}"

    def load_sample_raw_response(self) -> Dict[str, Any]:
        """Load a sample raw response from a JSON file."""
        try:
            file_path = os.path.join(SAMPLE_DIRECTORY, SAMPLE_FILE)
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading sample raw response: {str(e)}")
            return {"error": f"Error loading sample raw response: {str(e)}"}        
    
    def process_image(self, base64_image: str, image_name: str, image_index: int) -> Tuple[str, Dict[str, Any]]:
        print(f"process_image called in testing mode: {image_name = }")
        start_time = time.time()
        raw_response = None
        try:
            response_body = self.load_sample_raw_response()
            self.update_usage(response_body)
            text = self.extract_text(response_body)
            raw_response = self.save_raw_response(response_body, image_name)
            if self.include_random_error and random.random() < RANDOM_ERROR_THRESHOLD:
                raise Exception("Hypothetical Random Throttling Error Occurred")
            time_elapsed = (time.time() - start_time) / 60  # in minutes
            processing_data = self.get_transcript_processing_data(time_elapsed)
            self.num_processed += 1    
            return text, processing_data, raw_response
        except Exception as e:
            error_message = f"Error processing image: {str(e)}"
            print(error_message)
            return error_message, {"error": error_message}, raw_response
    
    def update_usage(self, response_data: Dict[str, Any]):
        """Update token usage from response data."""
        # Default implementation for models that don't have a specific usage extractor
        try:
            # Check for common usage structures
            if "usage" in response_data:
                usage = response_data["usage"]
                # Try different key formats
                if "input_tokens" in usage:
                    self.input_tokens = usage.get("input_tokens", 0)
                elif "inputTokens" in usage:
                    self.input_tokens = usage.get("inputTokens", 0)
                elif "prompt_tokens" in usage:
                    self.input_tokens = usage.get("prompt_tokens", 0)
                elif "inputTokenCount" in usage:
                    self.input_tokens = usage.get("inputTokenCount", 0)
                
                if "output_tokens" in usage:
                    self.output_tokens = usage.get("output_tokens", 0)
                elif "outputTokens" in usage:
                    self.output_tokens = usage.get("outputTokens", 0)
                elif "completion_tokens" in usage:
                    self.output_tokens = usage.get("completion_tokens", 0)
                elif "outputTokenCount" in usage:
                    self.output_tokens = usage.get("outputTokenCount", 0)
        except Exception as e:
            print(f"Error updating usage: {str(e)}")

########## End Testing Module  #############

class ClaudeImageProcessor(BedrockImageProcessor):
    """Specialized processor for Claude models."""
    
    def format_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Format the prompt for Claude models."""
        # Add a request for JSON output to the prompt text if not already present
        prompt_text = self.prompt_text
        if "json" not in prompt_text.lower():
            prompt_text += "\n\nPlease provide the transcription as a JSON object with a 'transcription' field."
        
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
        }
    
    def update_usage(self, response_data: Dict[str, Any]):
        """Update token usage from Claude response data."""
        usage = response_data.get("usage", {})
        self.input_tokens = usage.get("input_tokens", 0)
        self.output_tokens = usage.get("output_tokens", 0)
    
    def extract_text(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Claude response."""
        content = response_body.get("content", [])
        text = next((item.get("text", "") for item in content if item.get("type") == "text"), "")
        
        # Try to extract JSON from the text if it contains JSON markers
        if "{" in text and "}" in text:
            try:
                # Find JSON content between curly braces
                import re
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # Validate it's proper JSON by parsing it
                    json.loads(json_str)
                    # If successful, return just the JSON part
                    return json_str
            except json.JSONDecodeError:
                # If JSON parsing fails, return the full text
                pass
        
        return text


class NovaImageProcessor(BedrockImageProcessor):
    """Specialized processor for Amazon Nova models."""
    
    def format_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Format the prompt for Nova models."""
        # Add a request for JSON output to the prompt text if not already present
        prompt_text = self.prompt_text
        if "json" not in prompt_text.lower():
            prompt_text += "\n\nPlease provide the transcription as a JSON object with a 'transcription' field."
        
        # Based on debug_nova.py results, format 1 works for Nova models
        return {
            "schemaVersion": "messages-v1",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": base64_image},
                        }
                    },
                    {
                        "text": prompt_text
                    }
                ],
            }],
            "inferenceConfig": {"max_new_tokens": 4096, "top_p": 0.9, "temperature": 0.0}
        }
    
    def update_usage(self, response_data: Dict[str, Any]):
        """Update token usage from Nova response data."""
        usage = response_data.get("usage", {})
        self.input_tokens = usage.get("inputTokens", 0)
        self.output_tokens = usage.get("outputTokens", 0)
    
    def extract_text(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Nova response."""
        try:
            # Try to extract from the output.message.content structure
            if "output" in response_body and "message" in response_body["output"]:
                message = response_body["output"]["message"]
                if "content" in message and isinstance(message["content"], list):
                    for item in message["content"]:
                        if isinstance(item, dict) and "text" in item:
                            return item["text"]
            
            # Fall back to old format
            return response_body.get("results", [{}])[0].get("outputText", "")
        except Exception as e:
            # Import base64_filter here to avoid circular imports
            from utilities.base64_filter import filter_base64
            filtered_response = filter_base64(str(response_body))
            print(f"Error extracting text from Nova response: {str(e)}")
            return f"Error parsing response: {filtered_response[:500]}"


class AmazonImageProcessor(BedrockImageProcessor):
    """Specialized processor for other Amazon models (Titan, etc.)."""
    
    def format_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Format the prompt for Amazon models."""
        return {
            "inputText": self.prompt_text,
            "inputImage": base64_image,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "temperature": 0.0,
                "topP": 0.9
            }
        }
    
    def update_usage(self, response_data: Dict[str, Any]):
        """Update token usage from Amazon response data."""
        usage = response_data.get("usage", {})
        self.input_tokens = usage.get("inputTokenCount", 0)
        self.output_tokens = usage.get("outputTokenCount", 0)
    
    def extract_text(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Amazon response."""
        return response_body.get("results", [{}])[0].get("outputText", "")


class MetaImageProcessor(BedrockImageProcessor):
    """Specialized processor for Meta models."""
    
    def format_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Format the prompt for Meta models."""
        return {
            "prompt": self.prompt_text,
            "image": base64_image
        }
    
    def update_usage(self, response_data: Dict[str, Any]):
        """Update token usage from Meta response data."""
        usage = response_data.get("usage", {})
        self.input_tokens = usage.get("input_tokens", 0)
        self.output_tokens = usage.get("output_tokens", 0)
    
    def extract_text(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Meta response."""
        return response_body.get("generation", "") or response_body.get("text", "")


class MistralImageProcessor(BedrockImageProcessor):
    """Specialized processor for Mistral models."""
    
    def format_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Format the prompt for Mistral models."""
        return {
            "prompt": self.prompt_text,
            "image": base64_image,
            "max_tokens": 4096
        }
    
    def update_usage(self, response_data: Dict[str, Any]):
        """Update token usage from Mistral response data."""
        usage = response_data.get("usage", {})
        self.input_tokens = usage.get("prompt_tokens", 0)
        self.output_tokens = usage.get("completion_tokens", 0)
    
    def extract_text(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Mistral response."""
        return response_body.get("outputs", [{}])[0].get("text", "")


# Factory function to create the appropriate processor
def create_image_processor(api_key, prompt_name, prompt_text, model, modelname, output_name, testing=False):
    """Create the appropriate image processor based on the model."""
    provider = model.split(".")[0] if "." in model else ""
    # Load model info to check if it has an inference profile
    model_info = None
    try:
        with open("model_info/vision_model_info.json", "r") as f:
            models = json.load(f)
            for m in models:
                if m.get("modelId") == model:
                    model_info = m
                    break
    except Exception as e:
        print(f"Error loading model info: {str(e)}")
    # Create the appropriate processor based on provider
    try:
        if testing:
            return BedrockImageProcessorTesting(api_key, prompt_name, prompt_text, model, modelname, output_name)
        elif provider == "anthropic":
            return ClaudeImageProcessor(api_key, prompt_name, prompt_text, model, modelname, output_name)
        elif provider == "meta":
            return MetaImageProcessor(api_key, prompt_name, prompt_text, model, modelname, output_name)
        elif provider == "mistral":
            return MistralImageProcessor(api_key, prompt_name, prompt_text, model, modelname, output_name)
        elif provider == "amazon":
            if "nova" in model.lower():
                return NovaImageProcessor(api_key, prompt_name, prompt_text, model, modelname, output_name)
            else:
                return AmazonImageProcessor(api_key, prompt_name, prompt_text, model, modelname, output_name)
        else:
            # Default to base class for other providers
            print(f"Using default processor for model: {model} (provider: {provider})")
            return BedrockImageProcessor(api_key, prompt_name, prompt_text, model, modelname, output_name)
    except Exception as e:
        print(f"Error creating processor for model {model}: {str(e)}")
        # Fallback to base class if there's an error
        return BedrockImageProcessor(api_key, prompt_name, prompt_text, model, modelname, output_name)