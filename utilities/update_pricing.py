import json

# Load pricing data
with open('model_info/bedrock_models_pricing.json', 'r') as f:
    pricing_data = json.load(f)

# Update vision_model_info.json
with open('model_info/vision_model_info.json', 'r') as f:
    vision_models = json.load(f)

for model in vision_models:
    model_id = model.get('modelId', '')
    if '.' in model_id:
        provider = model_id.split('.')[0]
        model_name = model_id.split('.')[1].split(':')[0]
        
        if provider in pricing_data and model_name in pricing_data[provider]:
            price_info = pricing_data[provider][model_name]
            model['pricing'] = {
                'input': price_info['input_token_price_per_1M'],
                'output': price_info['output_token_price_per_1M']
            }
            model['pricing_verified'] = True
            print(f"Updated pricing for {model_id} in vision_model_info.json")
        else:
            model['pricing'] = {'input': 0.0, 'output': 0.0}
            model['pricing_verified'] = False

with open('model_info/vision_model_info.json', 'w') as f:
    json.dump(vision_models, f, indent=2)

# Update model_info.json
with open('model_info/model_info.json', 'r') as f:
    models = json.load(f)

for model in models:
    model_id = model.get('modelId', '')
    if '.' in model_id:
        provider = model_id.split('.')[0]
        model_name = model_id.split('.')[1].split(':')[0]
        
        if provider in pricing_data and model_name in pricing_data[provider]:
            price_info = pricing_data[provider][model_name]
            model['pricing'] = {
                'input': price_info['input_token_price_per_1M'],
                'output': price_info['output_token_price_per_1M']
            }
            model['pricing_verified'] = True
            print(f"Updated pricing for {model_id} in model_info.json")
        else:
            model['pricing'] = {'input': 0.0, 'output': 0.0}
            model['pricing_verified'] = False

with open('model_info/model_info.json', 'w') as f:
    json.dump(models, f, indent=2)

print("Pricing update completed!")