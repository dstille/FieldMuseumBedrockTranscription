import streamlit as st
import os
import json
import requests
import base64
from pathlib import Path
import time
import boto3
from bedrock_interface import BedrockImageProcessor

# Set page configuration
st.set_page_config(
    page_title="Bedrock Image Transcription App",
    page_icon="🖼️",
    layout="wide"
)

# Create necessary directories if they don't exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize directories
TEMP_IMAGES_DIR = "temp_images"
TRANSCRIPTIONS_DIR = "transcriptions"
RAW_RESPONSES_DIR = "raw_llm_responses"
PROMPTS_DIR = "prompts"

for directory in [TEMP_IMAGES_DIR, TRANSCRIPTIONS_DIR, RAW_RESPONSES_DIR]:
    ensure_directory_exists(directory)

# Load available models from selected_models.json
def load_models():
    try:
        with open("selected_models.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return []

# Load available prompts from the prompts folder
def load_prompts():
    prompts = {}
    try:
        for file in os.listdir(PROMPTS_DIR):
            if file.endswith(".txt"):
                file_path = os.path.join(PROMPTS_DIR, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                prompts[file] = content
        return prompts
    except Exception as e:
        st.error(f"Error loading prompts: {str(e)}")
        return {}

# Download image from URL and save to temp folder
def download_image(url, index):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Create a filename based on the URL
            filename = f"image_{index}_{int(time.time())}.jpg"
            file_path = os.path.join(TEMP_IMAGES_DIR, filename)
            
            # Save the image
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            return file_path, None
        else:
            return None, f"Failed to download image. Status code: {response.status_code}"
    except Exception as e:
        return None, f"Error downloading image: {str(e)}"

# Convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Save transcription to file
def save_transcription(content, image_name, model_name):
    timestamp = time.strftime("%Y-%m-%d-%H%M-%S")
    filename = f"{TRANSCRIPTIONS_DIR}/{image_name}_{model_name}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

# Main app
def main():
    st.title("Bedrock Image Transcription App")
    
    # Load models and prompts
    models = load_models()
    prompts = load_prompts()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Select prompt
        st.subheader("1. Select a Prompt")
        selected_prompt_name = st.selectbox("Choose a prompt:", list(prompts.keys()))
        selected_prompt_text = prompts[selected_prompt_name]
        
        # Display selected prompt
        with st.expander("View Selected Prompt"):
            st.text_area("Prompt Content", selected_prompt_text, height=200, disabled=True)
        
        # Upload URLs file
        st.subheader("2. Upload URLs File")
        uploaded_file = st.file_uploader("Upload a text file with URLs (one per line)", type=["txt"])
        
        # Select model
        st.subheader("3. Select Bedrock Model")
        selected_model = st.selectbox("Choose a model:", models)
        
        # Process button
        process_button = st.button("Process Images", type="primary")
    
    # Main content area
    if process_button and uploaded_file is not None:
        # Read URLs from the uploaded file
        urls = uploaded_file.getvalue().decode("utf-8").splitlines()
        urls = [url.strip() for url in urls if url.strip()]
        
        if not urls:
            st.error("No URLs found in the uploaded file.")
            return
        
        st.info(f"Processing {len(urls)} images...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each URL
        results = []
        for i, url in enumerate(urls):
            # Update progress
            progress = (i / len(urls))
            progress_bar.progress(progress)
            status_text.text(f"Processing image {i+1}/{len(urls)}: {url}")
            
            # Download image
            image_path, error = download_image(url, i)
            if error:
                results.append({"url": url, "status": "error", "message": error})
                continue
            
            # Get base64 of image
            base64_image = image_to_base64(image_path)
            
            # Extract image name from path
            image_name = Path(image_path).stem
            
            try:
                # Initialize the BedrockImageProcessor
                # Note: API key is assumed to be handled via environment variables
                # Extract model name from ID for display purposes
                model_name_parts = selected_model.split('.')
                if len(model_name_parts) > 1:
                    model_name = model_name_parts[1].split('-')[0]  # Get the base name (claude, llama, etc.)
                else:
                    model_name = selected_model
                    
                processor = BedrockImageProcessor(
                    api_key="",  # Empty as we're using AWS credentials from environment
                    prompt_name=selected_prompt_name,
                    prompt_text=selected_prompt_text,
                    model=selected_model,
                    modelname=model_name
                )
                
                # Process the image
                content, processing_data = processor.process_image(base64_image, image_name, i)
                
                # Save transcription
                transcription_path = save_transcription(content, image_name, processor.modelname)
                
                results.append({
                    "url": url,
                    "status": "success",
                    "transcription_path": transcription_path,
                    "processing_data": processing_data
                })
                
            except Exception as e:
                results.append({
                    "url": url,
                    "status": "error",
                    "message": f"Error processing image: {str(e)}"
                })
        
        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Display results
        st.header("Results")
        
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        
        st.write(f"Successfully processed: {success_count} images")
        st.write(f"Errors: {error_count} images")
        
        # Show detailed results
        for i, result in enumerate(results):
            with st.expander(f"Image {i+1}: {result['url']}"):
                if result["status"] == "success":
                    st.success("Successfully processed")
                    st.write(f"Transcription saved to: {result['transcription_path']}")
                    
                    # Display processing data
                    if result.get("processing_data"):
                        st.subheader("Processing Data")
                        for key, value in result["processing_data"].items():
                            st.write(f"{key}: {value}")
                    
                    # Display transcription content
                    with open(result["transcription_path"], "r", encoding="utf-8") as f:
                        transcription = f.read()
                    st.subheader("Transcription")
                    st.text_area("", transcription, height=200, disabled=True)
                else:
                    st.error(f"Error: {result['message']}")

if __name__ == "__main__":
    main()