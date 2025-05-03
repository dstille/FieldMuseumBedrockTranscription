import boto3
import json
import time
import base64
import re
import os
from llm_interface import ImageProcessor

class BedrockImageProcessor(ImageProcessor):
    def __init__(self, api_key, prompt_name, prompt_text, model="anthropic.claude-3-sonnet-20240229-v1:0", modelname="claude-3"):
        super().__init__(api_key, prompt_name, prompt_text, model, modelname)
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

    def set_token_costs_per_mil(self):
        # Set costs based on model - you'll need to update these based on AWS pricing
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
            "amazon.titan": {"input": 0.2, "output": 0.3}
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

    def _format_prompt(self, base64_image):
        """Format the prompt based on the model being used"""
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
        
        # Anthropic Claude models (all versions)
        if "anthropic.claude" in self.model:
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
        
        # Meta Llama models (all versions)
        elif "meta.llama" in self.model:
            return {
                "prompt": f"{structured_prompt}\n[IMAGE]{base64_image}[/IMAGE]",
                "max_gen_len": 4096,
                "temperature": 0,
                "top_p": 0.9,
                "response_format": {"type": "json_object"}
            }
        
        # Mistral models
        elif "mistral." in self.model:
            # Mistral models with image capability
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
        
        # Cohere models
        elif "cohere." in self.model:
            return {
                "message": structured_prompt,
                "max_tokens": 4096,
                "temperature": 0,
                "return_preamble": False,
                "stream": False,
                "response_format": {"type": "json_object"}
            }
        
        # AI21 models
        elif "ai21." in self.model:
            return {
                "prompt": structured_prompt,
                "maxTokens": 4096,
                "temperature": 0,
                "topP": 0.9,
                "stopSequences": [],
                "countPenalty": {"scale": 0},
                "presencePenalty": {"scale": 0},
                "frequencyPenalty": {"scale": 0}
            }
        
        # Amazon Titan models
        elif "amazon.titan" in self.model:
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
        
        # Default empty response for unsupported models
        return {}

    def process_image(self, base64_image, image_ref, index):
        start_time = time.time()
        try:
            # Check if the model supports image processing
            if not self._model_supports_images():
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

    def _model_supports_images(self):
        """Check if the model supports image processing"""
        # List of models that support image processing
        image_capable_models = [
            "anthropic.claude-3",  # All Claude 3 models
            "anthropic.claude-3-5", # Claude 3.5 models
            "anthropic.claude-3-7", # Claude 3.7 models
            "meta.llama3",  # Llama 3 models with vision
            "meta.llama4",  # Llama 4 models with vision
            "mistral.pixtral",  # Mistral's vision model
            "amazon.titan-image"  # Amazon Titan image models
        ]
        
        # Check if the model is in the list of image-capable models
        for model_prefix in image_capable_models:
            if model_prefix in self.model:
                return True
        
        return False

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

    def get_content_from_response(self, response_body):
        """Extract content from response based on model"""
        # Anthropic Claude models
        if "anthropic.claude" in self.model:
            return response_body.get("content", [{}])[0].get("text", "")
        
        # Meta Llama models
        elif "meta.llama" in self.model:
            return response_body.get("generation", "")
        
        # Mistral models
        elif "mistral." in self.model:
            return response_body.get("outputs", [{}])[0].get("text", "")
        
        # Cohere models
        elif "cohere." in self.model:
            return response_body.get("text", "")
        
        # AI21 models
        elif "ai21." in self.model:
            return response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
        
        # Amazon Titan text models
        elif "amazon.titan" in self.model and not "image" in self.model:
            return response_body.get("results", [{}])[0].get("outputText", "")
        
        # Amazon Titan image models
        elif "amazon.titan-image" in self.model:
            # For image generation models, we'd typically get back an image
            # But since we're using it for analysis, we should get text
            return response_body.get("text", "")
        
        # Default empty response for unsupported models
        return ""

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