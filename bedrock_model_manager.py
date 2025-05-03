import boto3
import os
import json

class BedrockModelManager:
    def __init__(self):
        self.bedrock = boto3.client(
            'bedrock',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.available_models = []
        self
        self.available_models_filename = 'available_models.json'
        self.selected_models_filename = 'selected_models.json'
        self.load_available_models()
        self.load_selected_models()

    def save_to_json(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            return json.load(f)    

    def display_models(self, models, msg):
        print(f"\n{msg}")
        print("========================")
        for model in models:
            print(model)            

    def load_available_models(self):
        if os.path.exists(self.available_models_filename):
            self.available_models = self.load_from_json(self.available_models_filename)
        else:
            self.available_models = self.get_available_models()
            self.save_to_json(self.available_models, self.available_models_filename)

    def load_selected_models(self):
        if os.path.exists(self.selected_models_filename):
            self.selected_models = self.load_from_json(self.selected_models_filename)
        else:
            self.selected_models = []
            self.save_to_json(self.selected_models, self.selected_models_filename)

    def get_available_models(self):
        try:
            all_models = self.bedrock.list_foundation_models()
            return [model['modelId'] for model in all_models['modelSummaries']]
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []

    def get_selected_models(self):
        return self.selected_models

    def modify_selected_models(self):
        self.display_models(self.available_models, "Available Models")
        self.display_models(self.selected_models, "Selected Models")
        models_to_add = []
        while True:
            model_id = input("\nEnter the model ID to add (or 'done' to finish): ")
            if model_id.lower() == 'done':
                break
            if model_id not in self.available_models:
                print("Invalid model ID. Please try again.")
                continue
            if model_id not in self.selected_models:
                self.selected_models.append(model_id)
            else:
                print("Model already in the list.")
        self.save_to_json(self.selected_models, self.selected_models_filename)    

if __name__ == "__main__":
    manager = BedrockModelManager()
    manager.modify_selected_models()
