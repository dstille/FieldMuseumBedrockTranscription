"""
Base image processor for AWS Bedrock models.
This module contains the base class for all model-specific processors.
"""

import json
import time
import os
import re
import boto3
from abc import ABC, abstractmethod

class BaseImageProcessor(ABC):
    """Base class for all AWS Bedrock image processors."""
    
    def __init__(self, api_key, prompt_name, prompt_text, model, modelname):
        """
        Initialize the base image processor.
        
        Args:
            api_key: API key (not used for AWS Bedrock, but kept for compatibility)
            prompt_name: Name of the prompt
            prompt_text: Text of the prompt
            model: Model ID
            modelname: Display name of the model
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
        # Convert text prompt to JSON if it's not already in JSON format
        self.prompt_json = self._ensure_json_prompt(prompt_text)
    
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
    
    def _ensure_json_prompt(self, prompt_text):
        """
        Ensure the prompt is in JSON format.
        If it's already JSON, parse it; otherwise convert text to JSON.
        """
        try:
            # Try to parse as JSON first
            if isinstance(prompt_text, dict):
                return prompt_text
            
            # Check if it's a JSON string
            return json.loads(prompt_text)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, convert text to JSON format
            return self._text_to_json_prompt(prompt_text)
    
    def _text_to_json_prompt(self, text):
        """Convert a text prompt to a structured JSON format."""
        # Initialize the result dictionary
        result = {
            "title": "",
            "description": "",
            "fields": []
        }
        
        # Extract the title and description (everything before the first field)
        lines = text.strip().split('\n')
        description_lines = []
        
        # Find where the fields start
        field_start_idx = 0
        for i, line in enumerate(lines):
            if re.search(r'^\s*\w+\s*:', line):
                field_start_idx = i
                break
            description_lines.append(line)
        
        # Set the title and description
        if description_lines:
            result["title"] = description_lines[0].strip()
            if len(description_lines) > 1:
                result["description"] = '\n'.join(description_lines[1:]).strip()
        
        # Process fields
        current_field = None
        current_description = []
        
        for i in range(field_start_idx, len(lines)):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this line starts a new field
            field_match = re.match(r'^(\w+)\s*:\s*(.*)', line)
            if field_match:
                # Save the previous field if it exists
                if current_field:
                    result["fields"].append({
                        "name": current_field,
                        "description": '\n'.join(current_description).strip()
                    })
                
                # Start a new field
                current_field = field_match.group(1)
                current_description = [field_match.group(2)]
            else:
                # Continue with the current field description
                if current_field:
                    current_description.append(line)
        
        # Add the last field
        if current_field:
            result["fields"].append({
                "name": current_field,
                "description": '\n'.join(current_description).strip()
            })
        
        return result
    
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
            return {"raw_text": text, "error": "Could not parse as JSON"}
    
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
                print(f"ERROR: {error_message}")
                return error_message, None
            
            request_body = self._format_prompt(base64_image)
            
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
    
    @abstractmethod
    def supports_image_processing(self):
        """Check if the model supports image processing."""
        pass
    
    @abstractmethod
    def _format_prompt(self, base64_image):
        """Format the prompt for the specific model."""
        pass
    
    @abstractmethod
    def get_content_from_response(self, response_body):
        """Extract content from the model's response."""
        pass
    
    @abstractmethod
    def set_token_costs_per_mil(self):
        """Set token costs per million tokens."""
        pass
    
    @abstractmethod
    def update_usage(self, response_data):
        """Update token usage from response data."""
        pass