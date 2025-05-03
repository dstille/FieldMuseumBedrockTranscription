#!/usr/bin/env python3
"""
Setup script for Field Museum Bedrock Transcription application.
This script creates necessary folders, installs requirements, and sets up .gitignore.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Application directories
DIRECTORIES = [
    "temp_images",
    "transcriptions",
    "raw_llm_responses",
    "prompts",
    "logs",
    "data"
]

# Requirements for the application
REQUIREMENTS = [
    "streamlit",
    "boto3",
    "requests",
    "python-dotenv",
    "pillow"
]

def print_header(message):
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}\n")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def create_directories():
    """Create all necessary directories for the application"""
    print_header("Creating Application Directories")
    
    for directory in DIRECTORIES:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print_success(f"Created directory: {directory}")
            else:
                print_warning(f"Directory already exists: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory {directory}: {str(e)}")
    
    # Create a sample prompt file if prompts directory is empty
    prompts_dir = Path("prompts")
    if not any(prompts_dir.iterdir()) if prompts_dir.exists() else False:
        sample_prompt_path = prompts_dir / "sample_prompt.txt"
        with open(sample_prompt_path, "w", encoding="utf-8") as f:
            f.write("Please transcribe all text visible in this image. Include any handwritten notes, typed text, labels, and captions.\n\n"
                    "Format your response as plain text, preserving the layout as much as possible.\n\n"
                    "If there are any parts that are illegible or uncertain, indicate this with [illegible].")
        print_success("Created sample prompt file")

def create_gitignore():
    """Create or update .gitignore file"""
    print_header("Setting up .gitignore")
    
    gitignore_content = """# Application specific folders
temp_images/
transcriptions/
raw_llm_responses/
logs/

# Environment variables
.env
.env.local

# Python cache files
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db
"""
    
    try:
        with open(".gitignore", "w", encoding="utf-8") as f:
            f.write(gitignore_content)
        print_success("Created .gitignore file")
    except Exception as e:
        print_error(f"Failed to create .gitignore file: {str(e)}")

def create_env_template():
    """Create a template .env file"""
    print_header("Creating .env template")
    
    env_content = """# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# Application Settings
DEBUG=False
"""
    
    try:
        if not os.path.exists(".env"):
            with open(".env", "w", encoding="utf-8") as f:
                f.write(env_content)
            print_success("Created .env template file")
        else:
            print_warning(".env file already exists, skipping")
    except Exception as e:
        print_error(f"Failed to create .env template: {str(e)}")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print_error(f"Python 3.8+ is required. You are using Python {major}.{minor}")
        return False
    
    print_success(f"Python version {major}.{minor} is compatible")
    return True

def install_requirements():
    """Install required Python packages"""
    print_header("Installing Requirements")
    
    # Determine pip command based on platform
    pip_cmd = "pip"
    if platform.system() == "Windows":
        pip_cmd = "pip"
    else:
        # Try to use pip3 on Unix-like systems
        try:
            subprocess.run(["pip3", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pip_cmd = "pip3"
        except:
            pip_cmd = "pip"
    
    # Install each requirement
    for package in REQUIREMENTS:
        try:
            print(f"Installing {package}...")
            subprocess.run([pip_cmd, "install", package], check=True)
            print_success(f"Installed {package}")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {package}: {str(e)}")
        except Exception as e:
            print_error(f"Error during installation of {package}: {str(e)}")

def create_selected_models_json():
    """Create selected_models.json if it doesn't exist"""
    print_header("Setting up model configuration")
    
    if not os.path.exists("selected_models.json"):
        models = [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "meta.llama2-70b-chat-v1"
        ]
        
        try:
            with open("selected_models.json", "w", encoding="utf-8") as f:
                json.dump(models, f, indent=2)
            print_success("Created selected_models.json")
        except Exception as e:
            print_error(f"Failed to create selected_models.json: {str(e)}")
    else:
        print_warning("selected_models.json already exists, skipping")

def main():
    """Main setup function"""
    print_header("Field Museum Bedrock Transcription App Setup")
    
    if not check_python_version():
        sys.exit(1)
    
    create_directories()
    create_gitignore()
    create_env_template()
    
    # Ask user if they want to install requirements
    install_req = input(f"{Colors.BLUE}Do you want to install Python requirements? (y/n): {Colors.ENDC}")
    if install_req.lower() == 'y':
        install_requirements()
    
    # Import json here to avoid issues if it's not available
    import json
    create_selected_models_json()
    
    print_header("Setup Complete!")
    print(f"""
{Colors.GREEN}Your Field Museum Bedrock Transcription App is ready to use!{Colors.ENDC}

To run the application:
1. Make sure to update your AWS credentials in the .env file
2. Run the app with: {Colors.BOLD}streamlit run app.py{Colors.ENDC}

{Colors.BLUE}Happy transcribing!{Colors.ENDC}
""")

if __name__ == "__main__":
    main()