"""
Amazon Titan image processor for AWS Bedrock.
This module contains the processor for Amazon Titan models.
"""

from .base_processor import BaseImageProcessor

class AmazonTitanImageProcessor(BaseImageProcessor):
    """Processor for Amazon Titan models."""
    
    def supports_image_processing(self):
        """Check if the model supports image processing."""
        # Only Titan Image Generator supports image processing, but it's for generation not analysis
        # For our use case, we'll say no Titan models support image analysis
        return False
    
    def set_token_costs_per_mil(self):
        """Set token costs per million tokens."""
        # Set costs based on model
        model_costs = {
            "titan-text-express": {"input": 0.13, "output": 0.17},
            "titan-text-lite": {"input": 0.03, "output": 0.04},
            "titan-text-premier": {"input": 0.2, "output": 0.3},
            "titan-embed": {"input": 0.05, "output": 0.0},
            "titan-image-generator": {"input": 0.0, "output": 0.0}  # Different pricing model
        }
        
        # Find the matching model
        model_type = None
        for model_name in model_costs:
            if model_name in self.model:
                model_type = model_name
                break
        
        # Default to titan-text-express if no match
        if not model_type:
            model_type = "titan-text-express"
        
        self.input_cost_per_mil = model_costs[model_type]["input"]
        self.output_cost_per_mil = model_costs[model_type]["output"]
    
    def _format_prompt(self, base64_image):
        """Format the prompt for Amazon Titan models."""
        # Create a JSON instruction that asks for JSON output
        json_instruction = "Return your response as a valid JSON object with fields matching the requested information. Do not include any text outside the JSON structure."
        
        # Get the prompt text from the JSON structure
        prompt_description = self.prompt_json.get("description", "")
        fields = self.prompt_json.get("fields", [])
        
        # Build a structured prompt text
        structured_prompt = f"{prompt_description}\n\n{json_instruction}\n\nPlease provide the following information in JSON format:"
        
        # Add fields to the prompt
        for field in fields:
            field_name = field.get("name", "")
            field_desc = field.get("description", "")
            structured_prompt += f"\n- {field_name}: {field_desc}"
        
        # Titan Image Generator
        if "image-generator" in self.model:
            return {
                "textToImageParams": {
                    "text": structured_prompt,
                    "negativeText": "",
                    "numberOfImages": 1,
                    "width": 1024,
                    "height": 1024,
                    "cfgScale": 8.0,
                    "seed": 0
                }
            }
        # Titan Text models
        else:
            return {
                "inputText": structured_prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 4096,
                    "temperature": 0,
                    "topP": 0.9,
                    "stopSequences": []
                }
            }
    
    def get_content_from_response(self, response_body):
        """Extract content from Amazon Titan's response."""
        # For text models
        if "results" in response_body:
            return response_body.get("results", [{}])[0].get("outputText", "")
        # For image generator models
        elif "images" in response_body:
            # Not applicable for our use case, but included for completeness
            return "Image generation models cannot be used for text analysis."
        return ""
    
    def update_usage(self, response_data):
        """Update token usage from Amazon Titan's response data."""
        if 'usage' in response_data:
            usage = response_data['usage']
            self.input_tokens += usage.get('inputTokenCount', 0)
            self.output_tokens += usage.get('outputTokenCount', 0)