# this script will read all the data files in a directory and recalculate costs according to the file "model_info/bedrock_models_pricing.json" and resave that data to the same file,
# in essence, it just adjusts the costs and keeps everything else the same

import json
import os
import re

DATA_DIR = 'data'
PRICING_FILEPATH = 'model_info/bedrock_models_pricing.json'

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_model_and_family_name(model_id):
    family_name, model_name_long = model_id.split('.')
    
    if 'sonnet' in model_name_long:
        if '3-5-sonnet' in model_name_long:
            if 'v2' in model_name_long:
                return family_name, 'claude_3.5_sonnet_v2'
            else:
                return family_name, 'claude_3.5_sonnet'
        elif '3-7-sonnet' in model_name_long:
            return family_name, 'claude_3.7_sonnet'
    elif 'nova' in model_name_long:
        if 'premier' in model_name_long:
            return family_name, 'nova_premier'
        elif 'pro' in model_name_long:
            return family_name, 'nova_pro'
        elif 'lite' in model_name_long:
            return family_name, 'nova_lite'
    
    return family_name, model_name_long

def calculate_cost(input_tokens, output_tokens, pricing_info):
    input_cost = (input_tokens / 1_000_000) * pricing_info['input_token_price_per_1M']
    output_cost = (output_tokens / 1_000_000) * pricing_info['output_token_price_per_1M']
    return input_cost, output_cost, input_cost + output_cost

def main():
    pricing = load_json(PRICING_FILEPATH)
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_DIR, filename)
            data = load_json(filepath)
            
            model_id = data['model']['id']
            family_name, model_name = get_model_and_family_name(model_id)
            
            if family_name in pricing and model_name in pricing[family_name]:
                pricing_info = pricing[family_name][model_name]
                input_tokens = data['tokens']['input']
                output_tokens = data['tokens']['output']
                
                old_total_cost = data['costs']['total']
                input_cost, output_cost, total_cost = calculate_cost(input_tokens, output_tokens, pricing_info)
                
                data['costs']['input'] = input_cost
                data['costs']['output'] = output_cost
                data['costs']['total'] = total_cost
                
                for image_url, image_data in data['images'].items():
                    if 'input tokens' in image_data and 'output tokens' in image_data:
                        img_input_cost, img_output_cost, _ = calculate_cost(
                            image_data['input tokens'], 
                            image_data['output tokens'], 
                            pricing_info
                        )
                        image_data['input cost $'] = round(img_input_cost, 3)
                        image_data['output cost $'] = round(img_output_cost, 3)
                
                save_json(data, filepath)
                print(f"Updated {filename}: ${old_total_cost:.4f} -> ${total_cost:.4f}")
            else:
                print(f"No pricing found for {model_id} in {filename}")

if __name__ == "__main__":
    main()