# this script will read all the data files in a directory and recalculate costs according to the file "model_info/bedrock_models_pricing.json" and resave that data to the same file,
# in essence, it just adjusts the costs and keeps everything else the same

import json
import os
import re
from dotenv import load_dotenv, set_key

DATA_DIR = 'data'
PRICING_FILEPATH = 'model_info/bedrock_models_pricing.json'

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model_and_family_name(model_id):
    family_name, model_name_long = model_id.split('.')
    # Remove version suffix (e.g., ':0') from model name
    model_name_clean = model_name_long.split(':')[0]
    return family_name, model_name_clean


def calculate_cost(input_tokens, output_tokens, pricing_info):
    input_cost = (input_tokens / 1_000_000) * pricing_info['input_token_price_per_1M']
    output_cost = (output_tokens / 1_000_000) * pricing_info['output_token_price_per_1M']
    return input_cost, output_cost, input_cost + output_cost

def main():
    load_dotenv()
    additional_flags_to_set = ['INCLUDE_STACK_TRACE', 'TESTING_MODE', 'INCLUDE_RANDOM_ERROR']
    # set each flag to false if they are not already in the .env
    for flag in additional_flags_to_set:
        if os.getenv(flag) is None:
            set_key('.env', flag, 'False')
    
    # Check if adjustment has already been run
    if os.getenv('COST_ADJUST_23_JUL_25', 'False').lower() == 'true':
        print("Cost adjustment has already been run.")
        return
    
    pricing = load_json(PRICING_FILEPATH)
    adjustment_made = False
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_DIR, filename)
            data = load_json(filepath)
            model_id = data['model']
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
                adjustment_made = True
            else:
                print(f"No pricing found for {model_id} in {filename}")
    
    if adjustment_made:
        # Set environment variable to indicate adjustment has been run
        set_key('.env', 'COST_ADJUST_23_JUL_25', 'True')
        print("Cost adjustment completed and flag set.")
    else:
        print("No cost adjustments were needed.")

if __name__ == "__main__":
    main()