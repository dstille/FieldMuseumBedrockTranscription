import os

def update_credentials():
    env_path = '.env'
    
    print("Current AWS credentials setup:")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'AWS_ACCESS_KEY_ID' in line or 'AWS_SECRET_ACCESS_KEY' in line or 'AWS_REGION' in line:
                    key = line.split('=')[0] if '=' in line else line
                    print(f"  {key}: {'SET' if '=' in line and line.split('=')[1].strip() != 'your_access_key_here' and line.split('=')[1].strip() != 'your_secret_key_here' else 'NOT SET'}")
    
    print("\nEnter new AWS credentials:")
    access_key = input("AWS Access Key ID: ").strip()
    secret_key = input("AWS Secret Access Key: ").strip()
    region = input("AWS Region (default: us-east-1): ").strip() or "us-east-1"
    
    env_content = f"""# AWS Credentials
AWS_ACCESS_KEY_ID={access_key}
AWS_SECRET_ACCESS_KEY={secret_key}
AWS_REGION={region}

# Application Settings
DEBUG=False
COST_ADJUST_10_JUL_25='true'
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\nCredentials updated in {env_path}")

if __name__ == "__main__":
    update_credentials()