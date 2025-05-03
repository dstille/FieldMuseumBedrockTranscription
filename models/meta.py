"""
Meta Llama image processor for AWS Bedrock.
This module contains the processor for Llama models.
"""

from .base_processor import BaseImageProcessor

class MetaImageProcessor(BaseImageProcessor):
    """Processor for Meta Llama models."""
    
    def supports_image_processing(self):
        """Check if the model supports image processing."""
        # Only Llama 4 models support image processing
        return "llama4" in self.model
    
    def set_token_costs_per_mil(self):
        """Set token costs per million tokens."""
        # Set costs based on model
        model_costs = {
            "llama3-8b": {"input": 0.2, "output": 0.6},
            "llama3-70b": {"input": 0.7, "output": 0.9},
            "llama4-scout": {"input": 1.0, "output": 3.0},
            "llama4-maverick": {"input": 1.0, "output": 3.0}
        }
        
        # Find the matching model
        model_type = None
        for model_name in model_costs:
            if model_name in self.model:
                model_type = model_name
                break
        
        # Default to llama3-70b if no match
        if not model_type:
            model_type = "llama3-70b"
        
        self.input_cost_per_mil = model_costs[model_type]["input"]
        self.output_cost_per_mil = model_costs[model_type]["output"]
    
    def _format_prompt(self, base64_image):
        """Format the prompt for Llama models."""
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
        
        # For Llama 4 models with vision capability
        if "llama4" in self.model:
            return {
                "prompt": f"{structured_prompt}\n[IMAGE]{base64_image}[/IMAGE]",
                "max_gen_len": 4096,
                "temperature": 0,
                "top_p": 0.9,
                "response_format": {"type": "json_object"}
            }
        # For text-only Llama models
        else:
            return {
                "prompt": structured_prompt,
                "max_gen_len": 4096,
                "temperature": 0,
                "top_p": 0.9,
                "response_format": {"type": "json_object"}
            }
    
    def get_content_from_response(self, response_body):
        """Extract content from Llama's response."""
        return response_body.get("generation", "")
    
    def update_usage(self, response_data):
        """Update token usage from Llama's response data."""
        if 'usage' in response_data:
            usage = response_data['usage']
            self.input_tokens += usage.get('prompt_tokens', 0)
            self.output_tokens += usage.get('completion_tokens', 0)