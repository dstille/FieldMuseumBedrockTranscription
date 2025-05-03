import boto3
import json
import time
import base64
from llm_interface import ImageProcessor
import os

class BedrockImageProcessor(ImageProcessor):
    def __init__(self, api_key, prompt_name, prompt_text, model="anthropic.claude-3-sonnet-20240229-v1:0", modelname="claude-3"):
        super().__init__(api_key, prompt_name, prompt_text, model, modelname)
        self.bedrock_client = self._initialize_bedrock_client()
        
    def _initialize_bedrock_client(self):
        """Initialize the Bedrock client with configuration"""
        # Fix: Use standard boto3 client configuration instead of boto3.Config
        return boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

    def set_token_costs_per_mil(self):
        # Set costs based on model - you'll need to update these based on AWS pricing
        model_costs = {
            "anthropic.claude-3": {"input": 15.0, "output": 75.0},
            "anthropic.claude-2": {"input": 8.0, "output": 24.0},
            "meta.llama2-70b": {"input": 0.7, "output": 0.9}
        }
        costs = model_costs.get(self.model.split(":")[0], {"input": 0, "output": 0})
        self.input_cost_per_mil = costs["input"]
        self.output_cost_per_mil = costs["output"]

    def _format_prompt(self, base64_image):
        """Format the prompt based on the model being used"""
        if "anthropic.claude" in self.model:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt_text
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
            }
        elif "meta.llama" in self.model:
            return {
                "prompt": f"{self.prompt_text}\n[IMAGE]{base64_image}[/IMAGE]",
                "max_gen_len": 2048,
                "temperature": 0,
                "top_p": 0.9
            }
        # Add more model-specific formatting as needed
        return {}

    def process_image(self, base64_image, image_ref, index):
        start_time = time.time()
        try:
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
            return content, self.get_transcript_processing_data(elapsed_time)
            
        except Exception as e:
            error_message = (
                f"Error processing image {index + 1} image '{image_ref}':\n {str(e)}"
            )
            print(f"ERROR: {error_message}")
            return error_message, None

    def get_content_from_response(self, response_body):
        """Extract content from response based on model"""
        if "anthropic.claude" in self.model:
            return response_body.get("content", [{}])[0].get("text", "")
        elif "meta.llama" in self.model:
            return response_body.get("generation", "")
        # Add more model-specific content extraction as needed
        return ""

    def update_usage(self, response_data):
        """Update token usage from response"""
        if 'usage' in response_data:
            usage = response_data['usage']
            self.input_tokens += usage.get('input_tokens', 0)
            self.output_tokens += usage.get('output_tokens', 0)