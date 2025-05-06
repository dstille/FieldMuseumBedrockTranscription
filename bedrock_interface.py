"""
Bedrock interface for image processing.
This module contains the processor for AWS Bedrock models.
"""

import json
import time
import os
import re
import base64
import boto3

class BedrockImageProcessor:
    """Processor for AWS Bedrock models."""
    
    def __init__(self, api_key, prompt_name, prompt_text, model, modelname, accept_non_json=False):
        """
        Initialize the Bedrock image processor.
        
        Args:
            api_key: API key (not used for AWS Bedrock, but kept for compatibility)
            prompt_name: Name of the prompt
            prompt_text: Text of the prompt
            model: Model ID
            modelname: Display name of the model
            accept_non_json: Whether to accept non-JSON responses
        """
        self.raw_response_folder = "raw_llm_responses"
        self.ensure_directory_exists(self.raw_response_folder)
        self.api_key = api_key.strip()
        self.prompt_name = prompt_name
        self.prompt_text = prompt_text
        self.model = model
        self.modelname = modelname
        self.input_tokens = 0
        self.output_tokens = 0
        self.set_token_costs_per_mil()
        self.num_processed = 0
        self.bedrock_client = self._initialize_bedrock_client()
        self.accept_non_json = accept_non_json
        
    def _initialize_bedrock_client(self):
        """Initialize the Bedrock client with configuration"""
        return boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def ensure_directory_exists(self, directory):
        """Create a directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def get_timestamp(self):
        """Get a formatted timestamp."""
        return time.strftime("%Y-%m-%d-%H%M-%S")
    
    def set_token_costs_per_mil(self):
        """Set token costs per million tokens."""
        # Set costs based on model
        model_costs = {
            "anthropic.claude-3": {"input": 15.0, "output": 75.0},
            "anthropic.claude-3-5": {"input": 3.0, "output": 15.0},
            "anthropic.claude-3-7": {"input": 5.0, "output": 15.0},
            "anthropic.claude-2": {"input": 8.0, "output": 24.0},
            "meta.llama3": {"input": 0.7, "output": 0.9},
            "meta.llama4": {"input": 1.0, "output": 3.0},
            "mistral.mistral": {"input": 0.5, "output": 1.5},
            "mistral.mixtral": {"input": 0.6, "output": 1.8},
            "cohere.command": {"input": 1.0, "output": 2.0},
            "ai21.jamba": {"input": 1.5, "output": 2.0},
            "amazon.titan": {"input": 0.2, "output": 0.3},
            "amazon.nova": {"input": 0.2, "output": 0.3},
            "us.amazon.nova": {"input": 0.2, "output": 0.3}
        }
        
        # Find the matching model family
        model_family = None
        for family in model_costs:
            if family in self.model:
                model_family = family
                break
        
        costs = model_costs.get(model_family, {"input": 0, "output": 0})
        self.input_cost_per_mil = costs["input"]
        self.output_cost_per_mil = costs["output"]
    
    def get_token_costs(self):
        """Calculate token costs."""
        return {
            "input tokens": self.input_tokens,
            "output tokens": self.output_tokens,
            "input cost $": round((self.input_tokens / 1_000_000) * self.input_cost_per_mil, 3),
            "output cost $": round((self.output_tokens / 1_000_000) * self.output_cost_per_mil, 3)
        }
    
    def get_transcript_processing_data(self, time_elapsed):
        """Get processing data for the transcript."""
        return {
            "time to create/edit (mins)": time_elapsed,
        } | self.get_token_costs()
    
    def save_raw_response(self, response_data, image_name):
        """Save the raw response to a file."""
        directory = f"{self.raw_response_folder}/{self.modelname}"
        self.ensure_directory_exists(directory)
        filename = f"{directory}/{image_name}-{self.get_timestamp()}-raw.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=4)
    
    def _extract_json_from_text(self, text):
        """
        Extract JSON from text response.
        Handles cases where the model might include text before or after the JSON.
        """
        try:
            # First try to parse the entire text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON using regex
            json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
            match = re.search(json_pattern, text)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return the original text as a JSON object
            if self.accept_non_json:
                return {"raw_text": text}
            else:
                return {"raw_text": text, "error": "Could not parse as JSON"}
    
    @staticmethod
    def get_verified_image_models():
        """
        Get a list of models that have been verified to work with images.
        This list is based on testing results and official documentation.
        
        Returns:
            A list of model IDs that support image processing
        """
        return [
            # Anthropic Claude models
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            
            # Amazon Nova models
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-v1:0",
            
            # Meta Llama models
            "meta.llama4-scout-17b-instruct-v1:0",
            "meta.llama4-maverick-17b-instruct-v1:0",
            
            # Mistral models
            "mistral.pixtral-large-2502-v1:0"
        ]
    
    @staticmethod
    def is_model_image_capable(model_id):
        """
        Check if a model supports image processing.
        
        Args:
            model_id: The model ID to check
            
        Returns:
            True if the model supports image processing, False otherwise
        """
        # List of model families that support image processing
        image_capable_families = [
            "anthropic.claude-3",  # All Claude 3 models
            "anthropic.claude-3-5", # Claude 3.5 models
            "anthropic.claude-3-7", # Claude 3.7 models
            "meta.llama4",  # Llama 4 models with vision
            "mistral.pixtral",  # Mistral's vision model
            "amazon.titan-image",  # Amazon Titan image models
            "amazon.nova",  # Amazon Nova models
            "us.amazon.nova"  # US Amazon Nova models
        ]
        
        # Get the list of verified image-capable models
        verified_models = BedrockImageProcessor.get_verified_image_models()
        
        # Check if the model is in the verified list
        if model_id in verified_models:
            return True
        
        # Check if the model belongs to a known image-capable family
        for family in image_capable_families:
            if family in model_id:
                return True
        
        return False
    
    def supports_image_processing(self):
        """Check if the model supports image processing."""
        return BedrockImageProcessor.is_model_image_capable(self.model)
    
    def process_image(self, base64_image, image_ref, index):
        """
        Process an image with the model.
        
        Args:
            base64_image: Base64-encoded image data
            image_ref: Reference to the image (e.g., filename)
            index: Index of the image in a batch
            
        Returns:
            Tuple of (content, processing_data)
        """
        start_time = time.time()
        try:
            # Check if the model supports image processing
            if not self.supports_image_processing():
                error_message = f"Model {self.model} does not support image processing"
                print(f"WARNING: {error_message}")
                # If we're testing all models, continue anyway
                if not self.accept_non_json:
                    return error_message, None
            
            # Format the prompt based on the model
            request_body = self._format_prompt(base64_image)
            
            # Debug the request body
            print(f"Using model: {self.model}")
            
            # Invoke the model
            response = self.bedrock_client.invoke_model(
                modelId=self.model,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response.get('body').read())
            self.save_raw_response(response_body, image_ref)
            
            # Update usage based on response
            if 'usage' in response_body:
                self.update_usage(response_body)
            
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            
            content = self.get_content_from_response(response_body)
            
            # Try to parse the content as JSON
            json_content = self._extract_json_from_text(content)
            
            # Return the JSON string
            return json.dumps(json_content, indent=2, ensure_ascii=False), self.get_transcript_processing_data(elapsed_time)
            
        except Exception as e:
            error_message = (
                f"Error processing image {index + 1} image '{image_ref}':\n {str(e)}"
            )
            print(f"ERROR: {error_message}")
            return error_message, None
    
    def _format_prompt(self, base64_image):
        """Format the prompt based on the model being used"""
        # Create a JSON instruction that asks for JSON output
        json_instruction = "Return your response as a valid JSON object with fields matching the requested information. Do not include any text outside the JSON structure."
        
        # Build a structured prompt text
        structured_prompt = f"{self.prompt_text}\n\n{json_instruction}\n\nPlease provide the information in JSON format."
        
        # Anthropic Claude 3.5 models (different format)
        if "anthropic.claude-3-5" in self.model:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": structured_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
                # Note: No response_format for Claude 3.5
            }
        
        # Anthropic Claude 3.7 models (may have different format)
        elif "anthropic.claude-3-7" in self.model:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": structured_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
                # Note: No response_format for Claude 3.7 (similar to 3.5)
            }
        
        # Anthropic Claude 3 models (original versions)
        elif "anthropic.claude-3" in self.model:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": structured_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ],
                "response_format": {"type": "json_object"}
            }
        
        # Amazon Nova models
        elif "amazon.nova" in self.model or "us.amazon.nova" in self.model:
            # Define system prompt
            system_list = [{
                "text": structured_prompt
            }]

            # Define message list
            message_list = [{
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": base64_image},
                        }
                    },
                    {
                        "text": "Analyze this image and provide the requested information in JSON format."
                    }
                ],
            }]

            # Configure inference parameters
            inf_params = {"max_new_tokens": 4096, "top_p": 0.1, "top_k": 20, "temperature": 0}

            return {
                "schemaVersion": "messages-v1",
                "messages": message_list,
                "system": system_list,
                "inferenceConfig": inf_params,
            }
        
        # Meta Llama 4 models (vision capable)
        elif "meta.llama4" in self.model:
            return {
                "prompt": f"{structured_prompt}\n[IMAGE]{base64_image}[/IMAGE]",
                "max_gen_len": 4096,
                "temperature": 0,
                "top_p": 0.9,
                "response_format": {"type": "json_object"}
            }
        
        # Mistral Pixtral models
        elif "mistral.pixtral" in self.model:
            return {
                "prompt": f"<s>[INST] {structured_prompt} [/INST]",
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.9,
                "images": [base64_image]
            }
        
        # Default format for text-only models - just send the prompt without the image
        return {
            "prompt": structured_prompt,
            "max_tokens": 4096,
            "temperature": 0
        }
    
    def get_content_from_response(self, response_body):
        """Extract content from response based on model"""
        # Anthropic Claude models
        if "anthropic.claude" in self.model:
            return response_body.get("content", [{}])[0].get("text", "")
        
        # Amazon Nova models
        elif "amazon.nova" in self.model or "us.amazon.nova" in self.model:
            return response_body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        
        # Meta Llama models
        elif "meta.llama" in self.model:
            return response_body.get("generation", "")
        
        # Mistral models
        elif "mistral." in self.model:
            return response_body.get("outputs", [{}])[0].get("text", "")
        
        # AI21 models
        elif "ai21." in self.model:
            return response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
        
        # Cohere models
        elif "cohere." in self.model:
            return response_body.get("text", "")
        
        # Amazon Titan models
        elif "amazon.titan" in self.model:
            return response_body.get("results", [{}])[0].get("outputText", "")
        
        # Default - try to extract any text content
        for key, value in response_body.items():
            if isinstance(value, str) and value:
                return value
        
        # If all else fails, return the entire response as a string
        return json.dumps(response_body)
    
    def update_usage(self, response_data):
        """Update token usage from response"""
        if 'usage' in response_data:
            usage = response_data['usage']
            
            # Different models use different field names for token counts
            input_tokens = usage.get('input_tokens', 
                           usage.get('prompt_tokens',
                           usage.get('inputTokenCount', 0)))
            
            output_tokens = usage.get('output_tokens',
                            usage.get('completion_tokens',
                            usage.get('outputTokenCount', 0)))
            
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens