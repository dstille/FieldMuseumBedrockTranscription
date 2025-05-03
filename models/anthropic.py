"""
Anthropic Claude image processor for AWS Bedrock.
This module contains the processor for Claude models.
"""

from .base_processor import BaseImageProcessor

class AnthropicImageProcessor(BaseImageProcessor):
    """Processor for Anthropic Claude models."""
    
    def supports_image_processing(self):
        """Check if the model supports image processing."""
        # Claude 3 models support image processing
        return "claude-3" in self.model or "claude-3-5" in self.model or "claude-3-7" in self.model
    
    def set_token_costs_per_mil(self):
        """Set token costs per million tokens."""
        # Set costs based on model
        model_costs = {
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-7-sonnet": {"input": 5.0, "output": 15.0},
            "claude-2": {"input": 8.0, "output": 24.0},
            "claude-instant": {"input": 1.63, "output": 5.51}
        }
        
        # Find the matching model
        model_type = None
        for model_name in model_costs:
            if model_name in self.model:
                model_type = model_name
                break
        
        # Default to sonnet if no match
        if not model_type:
            model_type = "claude-3-sonnet"
        
        self.input_cost_per_mil = model_costs[model_type]["input"]
        self.output_cost_per_mil = model_costs[model_type]["output"]
    
    def _format_prompt(self, base64_image):
        """Format the prompt for Claude models."""
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
    
    def get_content_from_response(self, response_body):
        """Extract content from Claude's response."""
        return response_body.get("content", [{}])[0].get("text", "")
    
    def update_usage(self, response_data):
        """Update token usage from Claude's response data."""
        if 'usage' in response_data:
            usage = response_data['usage']
            self.input_tokens += usage.get('input_tokens', 0)
            self.output_tokens += usage.get('output_tokens', 0)