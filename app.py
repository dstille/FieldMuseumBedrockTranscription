import streamlit as st
import os
import json
import requests
import base64
from pathlib import Path
import time
import boto3
from bedrock_interface import BedrockImageProcessor

# Initialize session state for persistence between reruns
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = {}
if 'results' not in st.session_state:
    st.session_state.results = []
if 'save_clicked' not in st.session_state:
    st.session_state.save_clicked = False
if 'save_completed' not in st.session_state:
    st.session_state.save_completed = False
if 'output_file_path' not in st.session_state:
    st.session_state.output_file_path = ""
if 'cost_data_path' not in st.session_state:
    st.session_state.cost_data_path = ""
if 'cost_summary' not in st.session_state:
    st.session_state.cost_summary = {}
if 'file_extension' not in st.session_state:
    st.session_state.file_extension = ""
if 'mime_type' not in st.session_state:
    st.session_state.mime_type = ""
if 'volume_name' not in st.session_state:
    st.session_state.volume_name = ""
if 'output_format' not in st.session_state:
    st.session_state.output_format = ""
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = ""
if 'model_name' not in st.session_state:
    st.session_state.model_name = ""
if 'selected_prompt_name' not in st.session_state:
    st.session_state.selected_prompt_name = ""

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
DATA_DIR = "data"
MODEL_INFO_DIR = "model_info"
UPLOAD_IMAGES_DIR = "images_to_upload"

for directory in [TEMP_IMAGES_DIR, TRANSCRIPTIONS_DIR, RAW_RESPONSES_DIR, DATA_DIR, MODEL_INFO_DIR, UPLOAD_IMAGES_DIR]:
    ensure_directory_exists(directory)

# Load available models from vision_model_info.json
def load_models():
    try:
        # Try to load from model_info directory first
        try:
            with open("model_info/vision_model_info.json", "r") as f:
                models = json.load(f)
        except FileNotFoundError:
            # Fall back to root directory
            with open("vision_model_info.json", "r") as f:
                models = json.load(f)
        
        # Add a display name for each model
        for model in models:
            model_id = model.get("modelId", "")
            model_name = model.get("modelName", "")
            provider = model.get("provider", "")
            model["display_name"] = f"{model_name} ({provider})"
        return models
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


def get_timestamp():
    return time.strftime("%Y-%m-%d-%H%M")

def get_volume_name(model_name_short):
    return f"{model_name_short}-{get_timestamp()}"    
        
# Download image from URL and save to temp folder
def download_image(url, index):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Get original filename from URL
            original_filename = url.split("/")[-1]
            if not original_filename or original_filename == url:
                # If no filename in URL, create a generic one
                original_filename = f"image_{index}"
            
            # Use the original filename directly (without timestamp)
            filename = original_filename
            file_path = os.path.join(TEMP_IMAGES_DIR, filename)
            
            # If file already exists, add a suffix
            if os.path.exists(file_path):
                base_name, extension = os.path.splitext(filename)
                filename = f"{base_name}_{int(time.time())}{extension}"
                file_path = os.path.join(TEMP_IMAGES_DIR, filename)
            
            # Save the image
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            return file_path, None
        else:
            return None, f"Failed to download image. Status code: {response.status_code}"
    except Exception as e:
        return None, f"Error downloading image: {str(e)}"

# Copy a local image to the temp folder
def copy_local_image(source_path, index):
    try:
        # Get the original filename
        original_filename = os.path.basename(source_path)
        
        # Create destination path
        dest_path = os.path.join(TEMP_IMAGES_DIR, original_filename)
        
        # If file already exists, add a suffix
        if os.path.exists(dest_path):
            base_name, extension = os.path.splitext(original_filename)
            new_filename = f"{base_name}_{int(time.time())}{extension}"
            dest_path = os.path.join(TEMP_IMAGES_DIR, new_filename)
        
        # Copy the file
        import shutil
        shutil.copy2(source_path, dest_path)
        
        return dest_path, None
    except Exception as e:
        return None, f"Error copying image: {str(e)}"

# Convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Save transcriptions to a JSON file
def save_transcriptions_json(transcriptions, volume_name):
    # Ensure the transcriptions directory exists
    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    
    # Create the JSON filename
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.json"
    
    # Process transcriptions to ensure proper JSON structure
    processed_transcriptions = {}
    for key, value in transcriptions.items():
        # Check if we need to parse any string JSON
        if isinstance(value, dict) and "transcription" in value and isinstance(value["transcription"], str):
            try:
                # Try to parse the JSON string
                parsed_json = json.loads(value["transcription"])
                value["transcription"] = parsed_json
                print(f"Parsed JSON string in transcription for {key}")
            except json.JSONDecodeError:
                # If it's not valid JSON, leave it as is
                print(f"Could not parse JSON string in transcription for {key}")
        
        processed_transcriptions[key] = value
    
    # Save the processed transcriptions to the JSON file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(processed_transcriptions, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved JSON transcriptions to {filename}")
        
        # Verify the file was created
        if os.path.exists(filename):
            print(f"Verified file exists: {filename}, size: {os.path.getsize(filename)} bytes")
        else:
            print(f"WARNING: File was not created: {filename}")
    except Exception as e:
        print(f"Error saving JSON transcriptions: {str(e)}")
        st.error(f"Error saving JSON transcriptions: {str(e)}")
    
    return filename

# Save transcriptions to a TXT file
def save_transcriptions_txt(transcriptions, volume_name):
    # Ensure the transcriptions directory exists
    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    
    # Create the TXT filename
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.txt"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for i, (image_name, data) in enumerate(transcriptions.items()):
                # Add separator if not the first item
                if i > 0:
                    f.write("\n\n")
                
                # Add header separator
                f.write("=" * 50 + "\n")
                
                # Write image name
                f.write(f"imageName: {image_name}\n\n")
                
                # Extract transcription data
                transcription_data = {}
                if isinstance(data, dict) and "transcription" in data:
                    if isinstance(data["transcription"], dict):
                        transcription_data = data["transcription"]
                    else:
                        transcription_data = {"transcription": data["transcription"]}
                else:
                    transcription_data = data
                
                # Write each field
                for key, value in transcription_data.items():
                    # Skip the transcription key if it's a dictionary
                    if key == "transcription" and isinstance(value, dict):
                        continue
                    
                    # Format the value, handling None, empty strings, etc.
                    formatted_value = str(value) if value is not None else "N/A"
                    if formatted_value.strip() == "":
                        formatted_value = "N/A"
                    
                    f.write(f"{key}: {formatted_value}\n")
                
                # Add footer separator
                f.write("\n" + "=" * 50)
        
        print(f"Successfully saved TXT transcriptions to {filename}")
    except Exception as e:
        print(f"Error saving TXT transcriptions: {str(e)}")
        st.error(f"Error saving TXT transcriptions: {str(e)}")
    
    return filename

# Save transcriptions to a CSV file
def save_transcriptions_csv(transcriptions, volume_name):
    print(f"{transcriptions = }")
    import csv
    
    # Ensure the transcriptions directory exists
    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    
    # Create the CSV filename
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.csv"
    
    try:
        # First, collect all possible field names from all transcriptions
        fieldnames = ["image_name"]  # Start with image_name as the first field
        
        for image_name, data in transcriptions.items():
            # Check if there's a transcription field that contains a dictionary
            if isinstance(data, dict) and "transcription" in data:
                if isinstance(data["transcription"], dict):
                    # Add all keys from the inner dictionary
                    for key in data["transcription"].keys():
                        if key not in fieldnames:
                            fieldnames.append(key)
                else:
                    # If transcription is not a dict, add it as a single field
                    if "transcription" not in fieldnames:
                        fieldnames.append("transcription")
            else:
                # For any other keys in the top-level dictionary
                for key in data.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        
        # Write the CSV file
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for image_name, data in transcriptions.items():
                row = {"image_name": image_name}
                
                # Extract values from the transcription dictionary
                if isinstance(data, dict) and "transcription" in data:
                    if isinstance(data["transcription"], dict):
                        # Add all values from the inner dictionary
                        for key, value in data["transcription"].items():
                            row[key] = value
                    else:
                        # If transcription is not a dict, add it as a single field
                        row["transcription"] = data["transcription"]
                else:
                    # For any other keys in the top-level dictionary
                    for key, value in data.items():
                        row[key] = value
                
                writer.writerow(row)
        
        print(f"Successfully saved CSV transcriptions to {filename}")
    except Exception as e:
        print(f"Error saving CSV transcriptions: {str(e)}")
        st.error(f"Error saving CSV transcriptions: {str(e)}")
    
    return filename

# Save cost data to a JSON file
def save_cost_data(volume_name, model_id, model_name, results, prompt_name):
    import datetime
    
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Create the data filename
    filename = f"{DATA_DIR}/{volume_name}-data.json"
    
    print(f"Saving cost data to {filename}")
    
    # Calculate total tokens and costs
    total_input_tokens = 0
    total_output_tokens = 0
    total_input_cost = 0.0
    total_output_cost = 0.0
    total_time = 0.0
    
    # Extract processing data from successful results
    for result in results:
        if result["status"] == "success" and "processing_data" in result:
            data = result["processing_data"]
            total_input_tokens += data.get("input tokens", 0)
            total_output_tokens += data.get("output tokens", 0)
            total_input_cost += data.get("input cost $", 0.0)
            total_output_cost += data.get("output cost $", 0.0)
            total_time += data.get("time to create/edit (mins)", 0.0)
    
    # Create the cost data structure
    cost_data = {
        "run_id": volume_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "model": {
            "id": model_id,
            "name": model_name
        },
        "prompt": prompt_name,
        "images_processed": len([r for r in results if r["status"] == "success"]),
        "images_failed": len([r for r in results if r["status"] == "error"]),
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_input_tokens + total_output_tokens
        },
        "costs": {
            "input": total_input_cost,
            "output": total_output_cost,
            "total": total_input_cost + total_output_cost
        },
        "processing_time_minutes": total_time
    }
    
    # Save the cost data to the JSON file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved cost data to {filename}")
        
        # Verify the file was created
        if os.path.exists(filename):
            print(f"Verified cost data file exists: {filename}, size: {os.path.getsize(filename)} bytes")
        else:
            print(f"WARNING: Cost data file was not created: {filename}")
    except Exception as e:
        print(f"Error saving cost data: {str(e)}")
        st.error(f"Error saving cost data: {str(e)}")
    
    return filename, cost_data

def save_transcriptions_callback():
    print(f"Save callback triggered for: {st.session_state.volume_name}")
    output_file_path = ""
    file_extension = ""
    mime_type = ""
    
    # Process transcriptions to ensure proper JSON structure
    processed_transcriptions = {}
    for key, value in st.session_state.transcriptions.items():
        # Check if we need to parse any string JSON
        if isinstance(value, dict) and "transcription" in value and isinstance(value["transcription"], str):
            try:
                # Try to parse the JSON string
                parsed_json = json.loads(value["transcription"])
                value["transcription"] = parsed_json
                print(f"Parsed JSON string in transcription for {key}")
            except json.JSONDecodeError:
                # If it's not valid JSON, leave it as is
                print(f"Could not parse JSON string in transcription for {key}")
        
        processed_transcriptions[key] = value
    
    # Update the session state with processed transcriptions
    st.session_state.transcriptions = processed_transcriptions
    
    try:
        if st.session_state.output_format == "JSON":
            output_file_path = save_transcriptions_json(st.session_state.transcriptions, st.session_state.volume_name)
            mime_type = "application/json"
            file_extension = "json"
        elif st.session_state.output_format == "TXT":
            output_file_path = save_transcriptions_txt(st.session_state.transcriptions, st.session_state.volume_name)
            mime_type = "text/plain"
            file_extension = "txt"
        elif st.session_state.output_format == "CSV":
            output_file_path = save_transcriptions_csv(st.session_state.transcriptions, st.session_state.volume_name)
            mime_type = "text/csv"
            file_extension = "csv"
        
        # Save cost data
        cost_data_path, cost_summary = save_cost_data(
            st.session_state.volume_name, 
            st.session_state.selected_model, 
            st.session_state.model_name, 
            st.session_state.results, 
            st.session_state.selected_prompt_name
        )
        
        print(f"Files saved successfully: {output_file_path}, {cost_data_path}")
        
        # Store paths and data in session state
        st.session_state.output_file_path = output_file_path
        st.session_state.cost_data_path = cost_data_path
        st.session_state.cost_summary = cost_summary
        st.session_state.file_extension = file_extension
        st.session_state.mime_type = mime_type
        st.session_state.save_completed = True
        
        # Don't rerun the app, just set a flag to show success message
        st.session_state.show_save_success = True
    except Exception as e:
        print(f"Error in save_transcriptions_callback: {str(e)}")
        st.error(f"Error saving files: {str(e)}")
        st.session_state.show_save_error = str(e)

# Main app
def main():
    st.title("Bedrock Image Transcription App")
    
    # Add a reset button to clear session state
    if st.sidebar.button("Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("App reset successfully!")
        st.experimental_rerun()
    
    # Load models and prompts
    all_models = load_models()
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
        
        # Input method selection
        st.subheader("2. Select Input Method")
        input_method = st.radio(
            "Choose input method:",
            ["Upload URLs File", "Select Local Images"],
            help="Upload a text file with URLs or select images from your local images_to_upload folder"
        )
        
        uploaded_file = None
        selected_local_images = []
        
        if input_method == "Upload URLs File":
            uploaded_file = st.file_uploader("Upload a text file with URLs (one per line)", type=["txt"])
        else:
            # List available images in the upload directory
            available_images = []
            for file in os.listdir(UPLOAD_IMAGES_DIR):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    available_images.append(file)
            
            if not available_images:
                st.warning(f"No images found in the {UPLOAD_IMAGES_DIR} folder. Please add some images and refresh.")
            else:
                selected_local_images = st.multiselect(
                    "Select images to process:",
                    available_images,
                    help=f"Select one or more images from the {UPLOAD_IMAGES_DIR} folder"
                )
        
        # Filter models by image test success and exclude Mistral models
        image_test_success_models = [model for model in all_models 
                                    if model.get("image_test_success", False) 
                                    and not model.get("modelId", "").startswith("mistral.")]
        image_capable_models = [model for model in all_models 
                               if model.get("supports_image", False)
                               and not model.get("modelId", "").startswith("mistral.")]
        all_non_mistral_models = [model for model in all_models 
                                 if not model.get("modelId", "").startswith("mistral.")]
        
        # Show model selection options
        st.subheader("3. Select Bedrock Model")
        model_filter = st.radio(
            "Filter models by:",
            ["Models that passed image test", "All image-capable models", "All models"],
            index=0
        )
        
        if model_filter == "Models that passed image test":
            models_to_show = image_test_success_models
            st.success(f"Showing {len(image_test_success_models)} models that successfully processed images.")
        elif model_filter == "All image-capable models":
            models_to_show = image_capable_models
            st.info(f"Showing {len(image_capable_models)} models with image support.")
        else:
            models_to_show = all_non_mistral_models
            if len(image_capable_models) < len(all_non_mistral_models):
                st.warning(f"Only {len(image_capable_models)} of {len(all_non_mistral_models)} models support image processing.")
        
        # Add note about Mistral models being excluded
        mistral_count = len([m for m in all_models if m.get("modelId", "").startswith("mistral.")])
        if mistral_count > 0:
            st.info(f"Note: {mistral_count} Mistral models have been excluded to prevent terminal flooding issues.")
        
        # Create a list of model display names for the selectbox
        model_options = {model.get("display_name", model.get("modelId", "")): model for model in models_to_show}
        selected_model_name = st.selectbox("Choose a model:", list(model_options.keys()))
        
        # Get the selected model object
        selected_model_obj = model_options[selected_model_name]
        selected_model = selected_model_obj.get("modelId", "")
        
        # Show model details
        with st.expander("Model Details"):
            st.json({
                "Model ID": selected_model_obj.get("modelId", ""),
                "Provider": selected_model_obj.get("provider", ""),
                "Supports Image": selected_model_obj.get("supports_image", False),
                "Image Test Success": selected_model_obj.get("image_test_success", False),
                "Uses Inference Profile": selected_model_obj.get("use_inference_profile", False),
                "On-Demand Supported": selected_model_obj.get("on_demand_supported", False)
            })
        
        # Show warning if selected model doesn't support images
        if not selected_model_obj.get("supports_image", False):
            st.error(f"Warning: {selected_model_name} does not support image processing!")
        
        # Select output format
        st.subheader("4. Select Output File Format")
        output_format = st.radio(
            "Choose output format:",
            ["JSON", "CSV"],
            help="JSON: Structured data format\nCSV: Spreadsheet format"
        )
        # Process button - disable if no input is provided
        process_button_disabled = (input_method == "Upload URLs File" and uploaded_file is None) or \
                                 (input_method == "Select Local Images" and not selected_local_images)
        process_button = st.button("Process Images", type="primary", disabled=process_button_disabled)
    
    # Main content area
    if process_button:
        print("process_button is True")
        st.success("Processing started...")
        # Check if we have valid input
        has_valid_input = False
        urls = []
        local_image_paths = []
        
        if input_method == "Upload URLs File" and uploaded_file is not None:
            # Read URLs from the uploaded file
            urls = uploaded_file.getvalue().decode("utf-8").splitlines()
            urls = [url.strip() for url in urls if url.strip()]
            
            if not urls:
                st.error("No URLs found in the uploaded file.")
                return
            
            has_valid_input = True
            st.info(f"Processing {len(urls)} images from URLs...")
        
        elif input_method == "Select Local Images" and selected_local_images:
            # Process selected local images
            local_image_paths = [os.path.join(UPLOAD_IMAGES_DIR, img) for img in selected_local_images]
            has_valid_input = True
            st.info(f"Processing {len(local_image_paths)} local images...")
        
        if not has_valid_input:
            st.error("Please provide input images (either upload a URL file or select local images).")
            return
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract model name for display purposes
        model_name_parts = selected_model.split('.')
        if len(model_name_parts) > 1:
            model_name = model_name_parts[1].split('-')[0]  # Get the base name (claude, llama, etc.)
        else:
            model_name = selected_model
        ##################
        # Dictionary to store all transcriptions
        transcriptions = {}
        results = []

        # Define the common image processing function
        def process_single_image(img_path, orig_filename, item_index, source_identifier):
            print("process_single_image called")
            st.success(f"Processing image: {orig_filename}")
            try:
                # Get base64 of image
                base64_image = image_to_base64(img_path)
                
                # Create processor for this model
                from bedrock_interface import create_image_processor
                processor = create_image_processor(
                    api_key="",  # Empty as we're using AWS credentials from environment
                    prompt_name=selected_prompt_name,
                    prompt_text=selected_prompt_text,
                    model=selected_model,
                    modelname=model_name
                )
                
                # Check if model supports image processing
                if not processor.supports_image_processing():
                    error_msg = f"Model {selected_model} does not support image processing"
                    results.append({
                        "url": source_identifier,
                        "status": "error",
                        "message": error_msg
                    })
                    return False
                
                # Process the image
                content, processing_data = processor.process_image(base64_image, orig_filename, item_index)
                
                # Try to parse the content as JSON if it's a string
                try:
                    if isinstance(content, str):
                        # Check if the model supports JSON output
                        if selected_model_obj.get("supports_json_output", False):
                            # The model returns JSON directly
                            transcription_data = json.loads(content)
                            
                            # Special handling for Nova Lite double-escaped JSON
                            if "transcription" in transcription_data and isinstance(transcription_data["transcription"], str):
                                try:
                                    # Try to parse the inner JSON string
                                    inner_json = json.loads(transcription_data["transcription"])
                                    # Replace the string with the parsed JSON
                                    transcription_data["transcription"] = inner_json
                                    print(f"Successfully parsed inner JSON for {orig_filename}")
                                except json.JSONDecodeError:
                                    # If it's not valid JSON, leave it as is
                                    print(f"Warning: Could not parse inner JSON for {orig_filename}")
                            
                            # Also handle the case where the JSON is directly in the transcription field
                            elif isinstance(transcription_data, dict) and all(isinstance(k, str) for k in transcription_data.keys()):
                                # Check if this looks like a transcription object
                                if any(key in transcription_data for key in ["verbatimCollectors", "collectedBy", "recordNumber"]):
                                    # It's already a proper transcription object, wrap it
                                    transcription_data = {"transcription": transcription_data}
                        else:
                            # Convert text output to JSON format
                            transcription_data = {"transcription": content}
                    else:
                        transcription_data = content
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the content as plain text
                    transcription_data = {"transcription": content}
                
                # Add the transcription to the dictionary using the original filename
                transcriptions[orig_filename] = transcription_data
                
                results.append({
                    "url": source_identifier,
                    "status": "success",
                    "image_name": orig_filename,
                    "processing_data": processing_data
                })
                return True
            except Exception as e:
                # Create a more detailed error message
                from utilities.error_message import ErrorMessage
                error_obj = ErrorMessage.from_exception(e)
                error_msg = error_obj.get_truncated_message(500)
                
                # Add more context to the error message
                if "access denied" in str(e).lower():
                    error_msg += "\nAccess denied: You may not have permissions to use this model."
                elif "throttling" in str(e).lower():
                    error_msg += "\nThrottling error: The service is currently rate limiting requests."
                elif "timeout" in str(e).lower():
                    error_msg += "\nTimeout error: The request took too long to complete."
                elif "not found" in str(e).lower() and "endpoint" in str(e).lower():
                    error_msg += "\nEndpoint not found: The inference endpoint for this model may not be set up."
                elif "validation error" in str(e).lower():
                    error_msg += "\nValidation error: The request format may be incorrect for this model."
                elif "format_prompt" in str(e).lower():
                    error_msg += "\nFormat error: This model may not have a proper formatter implemented."
                elif "quota exceeded" in str(e).lower():
                    error_msg += "\nQuota exceeded: You have reached your usage limit for this model."
                
                results.append({
                    "url": source_identifier,
                    "status": "error",
                    "message": f"Error processing image: {error_msg}"
                })
                return False

        #################    
        # Process URL images
        for i, url in enumerate(urls):
            # Update progress
            total_items = len(urls) + len(local_image_paths)
            progress = (i / total_items) if total_items > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            
            # Download image
            image_path, error = download_image(url, i)
            if error:
                results.append({"url": url, "status": "error", "message": error})
                continue
            
            # Get the original filename from the URL
            original_filename = url.split("/")[-1]
            if not original_filename or original_filename == url:
                original_filename = f"image_{i}"
            
            # Process the image
            process_single_image(image_path, original_filename, i, url)
        
        # Process local images
        for i, local_path in enumerate(local_image_paths):
            # Update progress
            total_items = len(urls) + len(local_image_paths)
            progress = ((len(urls) + i) / total_items) if total_items > 0 else 0
            progress_bar.progress(progress)
            
            filename = os.path.basename(local_path)
            status_text.text(f"Processing local image {i+1}/{len(local_image_paths)}: {filename}")
            
            # Copy image to temp folder
            image_path, error = copy_local_image(local_path, i)
            if error:
                results.append({"url": filename, "status": "error", "message": error})
                continue
            
            # Process the image
            process_single_image(image_path, filename, len(urls) + i, filename)
            
        # This block is now handled in the process_single_image function
    
        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Store results and transcriptions in session state for persistence
        st.session_state.transcriptions = transcriptions
        st.session_state.results = results
        st.session_state.selected_model = selected_model
        st.session_state.model_name = model_name
        st.session_state.selected_prompt_name = selected_prompt_name
        
        # Display results
        st.header("Results")
        
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        
        st.write(f"Successfully processed: {success_count} images")
        st.write(f"Errors: {error_count} images")
        
        # Show detailed results
        for i, result in enumerate(results):
            # Get the filename for display (could be URL or local filename)
            display_name = result['url']
            if '/' in display_name:
                display_name = display_name.split("/")[-1]
            with st.expander(f"Image {i+1}: {display_name}"):
                if result["status"] == "success":
                    st.success("Successfully processed")
                    
                    # Display processing data
                    if result.get("processing_data"):
                        st.subheader("Processing Data")
                        for key, value in result["processing_data"].items():
                            st.write(f"{key}: {value}")
                    
                    # Display transcription content
                    image_name = result.get("image_name")
                    if image_name and image_name in transcriptions:
                        st.subheader("Transcription")
                        
                        # Process the transcription for display
                        display_data = transcriptions[image_name]
                        if isinstance(display_data, dict) and "transcription" in display_data:
                            if isinstance(display_data["transcription"], str):
                                try:
                                    # Try to parse the JSON string for display
                                    parsed_json = json.loads(display_data["transcription"])
                                    st.json({"transcription": parsed_json})
                                except json.JSONDecodeError:
                                    # If it's not valid JSON, display as is
                                    st.json(display_data)
                            else:
                                # Already parsed, display as is
                                st.json(display_data)
                        else:
                            # Not in expected format, display as is
                            st.json(display_data)
                            
                        st.caption(f"Original filename: {image_name}")
                else:
                    st.error(f"Error: {result['message']}")

# Save all transcriptions based on selected format
        if st.session_state.transcriptions:
            # Generate suggested volume name
            suggested_volume_name = get_volume_name(st.session_state.model_name)
            
            # Allow user to edit the volume name
            st.subheader("Save Transcriptions")
            st.write("Enter a name for your transcription file:")
            
            # Create two columns for the input and explanation
            col1, col2 = st.columns([3, 2])
            
            with col1:
                volume_name = st.text_input(
                    "Volume Name",
                    value=suggested_volume_name,
                    help="You can use the suggested name or enter your own",
                    key="volume_name_input"
                )
                st.session_state.volume_name = volume_name
            
            with col2:
                st.info(f"Suggested name: {suggested_volume_name}")
                st.write("The file will be saved in the 'transcriptions' folder.")
            
            # Store output format in session state
            st.session_state.output_format = output_format
            
            # Use a button with on_click callback
            save_btn = st.button("Save Transcriptions", key="save_button", on_click=save_transcriptions_callback)
            if save_btn:
                print("Save button clicked - this should trigger the callback")
            # After displaying the results, check if files were saved
        if "save_completed" in st.session_state and st.session_state.save_completed:
            if "show_save_success" in st.session_state and st.session_state.show_save_success:
                st.success(f"All transcriptions saved to: {st.session_state.output_file_path}")
                st.success(f"Cost data saved to: {st.session_state.cost_data_path}")
                # Clear the flag so the message doesn't show on every rerun
                st.session_state.show_save_success = False
            
            # Display cost summary
            st.subheader("Cost Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Cost", f"${st.session_state.cost_summary['costs']['total']:.4f}")
            
            with col2:
                st.metric("Total Tokens", f"{st.session_state.cost_summary['tokens']['total']:,}")
            
            with col3:
                st.metric("Processing Time", f"{st.session_state.cost_summary['processing_time_minutes']:.2f} min")
            
            # Add download buttons
            if os.path.exists(st.session_state.output_file_path):
                with open(st.session_state.output_file_path, "r", encoding="utf-8") as f:
                    file_data = f.read()
                
                st.download_button(
                    label=f"Download Transcriptions ({st.session_state.output_format})",
                    data=file_data,
                    file_name=f"{st.session_state.volume_name}.{st.session_state.file_extension}",
                    mime=st.session_state.mime_type,
                    key="transcription_download"
                )
            
            if os.path.exists(st.session_state.cost_data_path):
                with open(st.session_state.cost_data_path, "r", encoding="utf-8") as f:
                    cost_data = f.read()
                
                st.download_button(
                    label="Download Cost Data (JSON)",
                    data=cost_data,
                    file_name=f"{st.session_state.volume_name}-data.json",
                    mime="application/json",
                    key="cost_data_download"
                )

if __name__ == "__main__":
    main()