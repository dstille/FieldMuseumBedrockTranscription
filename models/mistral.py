"""
Mistral image processor for AWS Bedrock.
This module contains the processor for Mistral models.
"""

from .base_processor import BaseImageProcessor

class MistralImageProcessor(BaseImageProcessor):
    """Processor for Mistral models."""
    
    def supports_image_processing(self):
        """Check if the model supports image processing."""
        # Only Pixtral models support image processing
        return "pixtral" in self.model
    
    def set_token_costs_per_mil(self):
        """Set token costs per million tokens."""
        # Set costs based on model
        model_costs = {
            "mistral-7b": {"input": 0.2, "output": 0.6},
            "mistral-large": {"input": 0.8, "output": 2.4},
            "mistral-small": {"input": 0.2, "output": 0.6},
            "mixtral-8x7b": {"input": 0.6, "output": 1.8},
            "pixtral-large": {"input": 1.0, "output": 3.0}
        }
        
        # Find the matching model
        model_type = None
        for model_name in model_costs:
            if model_name in self.model:
                model_type = model_name
                break
        
        # Default to mistral-large if no match
        if not model_type:
            model_type = "mistral-large"
        
        self.input_cost_per_mil = model_costs[model_type]["input"]
        self.output_cost_per_mil = model_costs[model_type]["output"]
    
    def _format_prompt(self, base64_image):
        """Format the prompt for Mistral models."""
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
        
        # Mistral models with image capability (Pixtral)
        if "pixtral" in self.model:
            return {
                "prompt": f"<s>[INST] {structured_prompt} [/INST]",
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.9,
                "images": [base64_image]
            }
        # Text-only Mistral models
        else:
            return {
                "prompt": f"<s>[INST] {structured_prompt} [/INST]",
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.9
            }
    
    def get_content_from_response(self, response_body):
        """Extract content from Mistral's response."""
        return response_body.get("outputs", [{}])[0].get("text", "")
    
    def update_usage(self, response_data):
        """Update token usage from Mistral's response data."""
        if 'usage' in response_data:
            usage = response_data['usage']
            self.input_tokens += usage.get('input_tokens', 0)
            self.output_tokens += usage.get('output_tokens', 0)