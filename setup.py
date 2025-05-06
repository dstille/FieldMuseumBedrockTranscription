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
import venv
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
    "data",
    "test_images",
    "test_results"
]

# Requirements for the application
REQUIREMENTS = [
    "streamlit",
    "boto3",
    "requests",
    "python-dotenv",
    "pillow",
    "tabulate"
]

# Virtual environment name
VENV_NAME = "venv"

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

def create_virtual_environment():
    """Create a virtual environment"""
    print_header("Creating Virtual Environment")
    
    if os.path.exists(VENV_NAME):
        print_warning(f"Virtual environment '{VENV_NAME}' already exists")
        recreate = input(f"{Colors.BLUE}Do you want to recreate it? (y/n): {Colors.ENDC}")
        if recreate.lower() == 'y':
            try:
                shutil.rmtree(VENV_NAME)
                print_success(f"Removed existing virtual environment: {VENV_NAME}")
            except Exception as e:
                print_error(f"Failed to remove existing virtual environment: {str(e)}")
                return False
        else:
            return True
    
    try:
        venv.create(VENV_NAME, with_pip=True)
        print_success(f"Created virtual environment: {VENV_NAME}")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {str(e)}")
        return False

def get_venv_python_path():
    """Get the path to the Python executable in the virtual environment"""
    if platform.system() == "Windows":
        return os.path.join(VENV_NAME, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_NAME, "bin", "python")

def get_venv_pip_path():
    """Get the path to the pip executable in the virtual environment"""
    if platform.system() == "Windows":
        return os.path.join(VENV_NAME, "Scripts", "pip.exe")
    else:
        return os.path.join(VENV_NAME, "bin", "pip")

def install_requirements_in_venv():
    """Install required Python packages in the virtual environment"""
    print_header("Installing Requirements in Virtual Environment")
    
    pip_path = get_venv_pip_path()
    
    # Upgrade pip first
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print_success("Upgraded pip to the latest version")
    except Exception as e:
        print_warning(f"Failed to upgrade pip: {str(e)}")
    
    # Install each requirement
    for package in REQUIREMENTS:
        try:
            print(f"Installing {package}...")
            subprocess.run([pip_path, "install", package], check=True)
            print_success(f"Installed {package}")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {package}: {str(e)}")
        except Exception as e:
            print_error(f"Error during installation of {package}: {str(e)}")
    
    # Create requirements.txt file
    try:
        subprocess.run([pip_path, "freeze"], stdout=open("requirements.txt", "w"), check=True)
        print_success("Created requirements.txt file")
    except Exception as e:
        print_error(f"Failed to create requirements.txt: {str(e)}")

def create_selected_models_json():
    """Create selected_models.json if it doesn't exist"""
    print_header("Setting up model configuration")
    
    if not os.path.exists("selected_models.json"):
        models = [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-v1:0"
        ]
        
        try:
            import json
            with open("selected_models.json", "w", encoding="utf-8") as f:
                json.dump(models, f, indent=2)
            print_success("Created selected_models.json")
        except Exception as e:
            print_error(f"Failed to create selected_models.json: {str(e)}")
    else:
        print_warning("selected_models.json already exists, skipping")

def create_activation_scripts():
    """Create activation scripts for the virtual environment"""
    print_header("Creating Activation Scripts")
    
    # Create Windows activation script
    if platform.system() == "Windows":
        with open("activate.bat", "w") as f:
            f.write(f"@echo off\n"
                   f"echo Activating virtual environment...\n"
                   f"call {VENV_NAME}\\Scripts\\activate.bat\n"
                   f"echo Virtual environment activated. Type 'deactivate' to exit.\n")
        print_success("Created activate.bat for Windows")
    
    # Create Unix activation script
    else:
        with open("activate.sh", "w") as f:
            f.write(f"#!/bin/bash\n"
                   f"echo Activating virtual environment...\n"
                   f"source {VENV_NAME}/bin/activate\n"
                   f"echo Virtual environment activated. Type 'deactivate' to exit.\n")
        os.chmod("activate.sh", 0o755)  # Make executable
        print_success("Created activate.sh for Unix/Linux/Mac")

def main():
    """Main setup function"""
    print_header("Field Museum Bedrock Transcription App Setup")
    
    if not check_python_version():
        sys.exit(1)
    
    create_directories()
    create_gitignore()
    create_env_template()
    
    # Ask user if they want to create a virtual environment
    create_venv = input(f"{Colors.BLUE}Do you want to create a virtual environment? (y/n): {Colors.ENDC}")
    if create_venv.lower() == 'y':
        if create_virtual_environment():
            install_requirements_in_venv()
            create_activation_scripts()
    else:
        # Ask user if they want to install requirements globally
        install_req = input(f"{Colors.BLUE}Do you want to install Python requirements globally? (y/n): {Colors.ENDC}")
        if install_req.lower() == 'y':
            # Determine pip command based on platform
            pip_cmd = "pip"
            if platform.system() != "Windows":
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
                except Exception as e:
                    print_error(f"Error during installation of {package}: {str(e)}")
    
    # Import json here to avoid issues if it's not available
    import json
    create_selected_models_json()
    
    print_header("Setup Complete!")
    
    if create_venv.lower() == 'y':
        if platform.system() == "Windows":
            activate_cmd = "activate.bat"
        else:
            activate_cmd = "source activate.sh"
        
        print(f"""
{Colors.GREEN}Your Field Museum Bedrock Transcription App is ready to use!{Colors.ENDC}

To activate the virtual environment:
- Windows: Run {Colors.BOLD}activate.bat{Colors.ENDC}
- Unix/Linux/Mac: Run {Colors.BOLD}source activate.sh{Colors.ENDC}

After activating the virtual environment:
1. Make sure to update your AWS credentials in the .env file
2. Run the app with: {Colors.BOLD}streamlit run app.py{Colors.ENDC}

{Colors.BLUE}Happy transcribing!{Colors.ENDC}
""")
    else:
        print(f"""
{Colors.GREEN}Your Field Museum Bedrock Transcription App is ready to use!{Colors.ENDC}

To run the application:
1. Make sure to update your AWS credentials in the .env file
2. Run the app with: {Colors.BOLD}streamlit run app.py{Colors.ENDC}

{Colors.BLUE}Happy transcribing!{Colors.ENDC}
""")

if __name__ == "__main__":
    main()