from PIL import Image
import io
import time
import json
import os
import re
from utilities import utils

class ImageProcessor:

    def __init__(self, api_key, prompt_name, prompt_text, model, modelname, output_name):
        raw_response_dir = "raw_llm_responses"
        self.ensure_directory_exists(raw_response_dir)
        self.output_name = output_name
        self.raw_response_folder = os.path.join(raw_response_dir, output_name)
        self.ensure_directory_exists(self.raw_response_folder)
        self.api_key = api_key.strip()
        self.prompt_name = prompt_name
        self.prompt_text = prompt_text
        self.fieldnames = self.get_fieldnames()
        self.model = model
        self.modelname = modelname
        self.input_tokens = 0
        self.output_tokens = 0
        self.pricing_data = self.load_pricing_data()
        self.set_token_costs_per_mil()
        self.num_processed = 0

    def get_fieldnames(self):
        fieldnames =  utils.get_fieldnames_from_prompt_text(self.prompt_text)
        return fieldnames or "fieldnames not extracted"

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)    

    def get_timestamp(self):
        return  time.strftime("%Y-%m-%d-%H%M-%S")
    
    def get_token_costs(self):
        return {
            "input tokens": self.input_tokens,
            "output tokens": self.output_tokens,
            "input cost $": round((self.input_tokens / 1_000_000) * self.input_cost_per_mil, 3),
            "output cost $": round((self.output_tokens / 1_000_000) * self.output_cost_per_mil, 3)
        } 

    def load_pricing_data(self):
        """Load pricing data from the bedrock_models_pricing.json file."""
        try:
            path = "model_info/bedrock_models_pricing.json"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            print("Warning: Could not find model_info/bedrock_models_pricing.json file")
            return {}
        except Exception as e:
            print(f"Error loading pricing data: {str(e)}")
            return {}            

    def get_transcript_processing_data(self, time_elapsed):
        return {
                "time to create/edit (mins)": time_elapsed,
                } | self.get_token_costs()

    def get_legal_filename(self, filename):
        return re.sub(r'[\\/*?:]', "_", filename)

    def resize_image(self, image_bytes, max_size=(1120, 1120)):
        img = Image.open(io.BytesIO(image_bytes))
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format)
            return img_byte_arr.getvalue()
        return image_bytes         

    def save_raw_response(self, response_data, image_name):
        legal_image_name = self.get_legal_filename(image_name)
        filename = f"{self.raw_response_folder}/{legal_image_name}-raw.json"
        # Limit the size of the response data to avoid huge files
        max_size = 10000  # Maximum characters to save
        raw_response = None
        try:
            # Import base64_filter here to avoid circular imports
            from utilities.base64_filter import filter_base64_from_dict
            # Filter out base64 content before saving
            raw_response = filter_base64_from_dict(response_data)
            # Convert to string and check size
            response_str = json.dumps(raw_response, ensure_ascii=False, indent=4)
            if len(response_str) > max_size:
                # Create a truncated version
                raw_response = {"truncated_response": response_str[:max_size] + "..."}
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(raw_response, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved raw response to {filename}")
            return raw_response
        except Exception as e:
            print(f"Error saving raw response: {str(e)}")
            return raw_response

    def update_usage(self, response_data):
        if "usage" in response_data:
            usage = response_data["usage"]
            self.input_tokens = int(usage.get("prompt_tokens", 0))
            self.output_tokens = int(usage.get("completion_tokens", 0))                           

    def set_token_costs_per_mil(self):
        """Set token costs based on the model provider and model name from pricing data."""
        provider = self.model.split(".")[0] if "." in self.model else ""
        model_short_name = self.model.split(".")[-1].split(":")[0] if "." in self.model else self.model
        
        # Default costs
        self.input_cost_per_mil = 0.0
        self.output_cost_per_mil = 0.0
        
        # Try to get pricing from the pricing data file
        if hasattr(self, 'pricing_data') and self.pricing_data and provider in self.pricing_data:
            if model_short_name in self.pricing_data[provider]:
                price_info = self.pricing_data[provider][model_short_name]
                self.input_cost_per_mil = price_info.get("input_token_price_per_1M", self.input_cost_per_mil)
                self.output_cost_per_mil = price_info.get("output_token_price_per_1M", self.output_cost_per_mil)
                print(f"Found pricing for {model_short_name}: Input=${self.input_cost_per_mil}/1M, Output=${self.output_cost_per_mil}/1M")
            else:
                print(f"No pricing found for {provider}.{model_short_name}, using null values!!!!")
        