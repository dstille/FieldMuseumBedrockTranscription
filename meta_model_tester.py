#!/usr/bin/env python3
"""
Meta Model Tester - Debug version that only prints results without saving.
"""

import json
import base64
import time
from datetime import datetime

from bedrock_interface import create_image_processor

class MetaModelTester:
    def __init__(self, prompt_file="prompts/1.5Stripped.txt", test_image="testing/test_images/Test_Image.jpg"):
        self.prompt_text = self.load_prompt_text(prompt_file)
        self.base64_image = self.image_to_base64(test_image)
    
    def load_prompt_text(self, prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading prompt: {e}")
            return ""
    
    def image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image: {e}")
            return ""
    
    def load_meta_models(self):
        try:
            with open("model_info/vision_model_info.json", 'r') as f:
                models = json.load(f)
            return [m for m in models if m.get("provider") == "Meta"]
        except Exception as e:
            print(f"Error loading models: {e}")
            return []
    
    def test_model(self, model):
        model_id = model.get("modelId")
        model_name = model.get("modelName")
        
        print(f"\nTesting: {model_id} ({model_name})")
        
        try:
            processor = create_image_processor(
                api_key="",
                prompt_name="meta_test",
                prompt_text=self.prompt_text,
                model=model_id,
                modelname=model_name
            )
            
            start_time = time.time()
            text, processing_data, raw_response = processor.process_image(self.base64_image, "Test_Image", 0)
            elapsed_time = time.time() - start_time
            
            success = "error" not in processing_data and processing_data.get("input tokens", 0) > 0
            
            print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Tokens: {processing_data.get('input tokens', 0)} in, {processing_data.get('output tokens', 0)} out")
            print(f"  Cost: ${processing_data.get('total cost $', 0):.4f}")
            
            if success:
                print(f"  Response: {text[:200]}...")
            else:
                print(f"  Error: {processing_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  EXCEPTION: {str(e)}")
    
    def test_all_meta_models(self):
        models = self.load_meta_models()
        print(f"Found {len(models)} Meta models to test")
        
        for model in models:
            self.test_model(model)

if __name__ == "__main__":
    tester = MetaModelTester()
    tester.test_all_meta_models()