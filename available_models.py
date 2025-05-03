import boto3
import json
import os

def get_available_models():
    """
    Gets a list of available Bedrock models that support on-demand throughput
    """
    try:
        bedrock = boto3.client(
            'bedrock',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Get all models in region
        all_models = bedrock.list_foundation_models()
        
        # Filter for available models that support on-demand throughput
        available_models = []
        
        for model in all_models.get('modelSummaries', []):
            model_id = model.get('modelId', '')
            status = model.get('modelLifecycle', {}).get('status', '')
            
            # Only include models that are available
            if status == 'ACTIVE':
                available_models.append(model_id)
                
        # Save to selected_models.json
        with open('selected_models.json', 'w') as f:
            json.dump(available_models, f, indent=4)
            
        print(f"Found {len(available_models)} available models")
        print("Models saved to selected_models.json")
        
        return available_models
        
    except Exception as e:
        print(f"Error getting available models: {str(e)}")
        return []

if __name__ == "__main__":
    get_available_models()