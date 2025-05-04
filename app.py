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
DATA_DIR = "data"

for directory in [TEMP_IMAGES_DIR, TRANSCRIPTIONS_DIR, RAW_RESPONSES_DIR, DATA_DIR]:
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


def get_timestamp():
    return time.strftime("%Y-%m-%d-%H%M")

def get_volume_name(model_name_short):
    return f"{model_name_short}-{get_timestamp()}"    
        
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

# Save transcriptions to a JSON file
def save_transcriptions_json(transcriptions, volume_name):
    # Create the JSON filename
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.json"
    
    # Save the transcriptions to the JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, indent=2, ensure_ascii=False)
    
    return filename

# Save transcriptions to a TXT file
def save_transcriptions_txt(transcriptions, volume_name):
    # Create the TXT filename
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.txt"
    
    # Save the transcriptions to the TXT file
    with open(filename, "w", encoding="utf-8") as f:
        for image_name, data in transcriptions.items():
            f.write(f"=== {image_name} ===\n\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(str(data))
            
            f.write("\n\n" + "="*50 + "\n\n")
    
    return filename

# Save transcriptions to a CSV file
def save_transcriptions_csv(transcriptions, volume_name):
    import csv
    
    # Create the CSV filename
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.csv"
    
    # Get all possible field names from all transcriptions
    all_fields = set()
    for data in transcriptions.values():
        if isinstance(data, dict):
            all_fields.update(data.keys())
    
    # Convert set to sorted list for consistent column order
    field_names = sorted(list(all_fields))
    
    # Add image_name as the first column
    columns = ["image_name"] + field_names
    
    # Save the transcriptions to the CSV file
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for image_name, data in transcriptions.items():
            row = {"image_name": image_name}
            
            if isinstance(data, dict):
                for key, value in data.items():
                    # Handle nested structures by converting to string
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value)
                    else:
                        row[key] = value
            
            writer.writerow(row)
    
    return filename

# Save cost data to a JSON file
def save_cost_data(volume_name, model_id, model_name, results, prompt_name):
    import datetime
    
    # Create the data filename
    filename = f"{DATA_DIR}/{volume_name}-data.json"
    
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
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cost_data, f, indent=2, ensure_ascii=False)
    
    return filename, cost_data

# Main app
def main():
    st.title("Bedrock Image Transcription App")
    
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
        
        # Upload URLs file
        st.subheader("2. Upload URLs File")
        uploaded_file = st.file_uploader("Upload a text file with URLs (one per line)", type=["txt"])
        
        # Filter models by image capability
        image_capable_models = [model for model in all_models if BedrockImageProcessor.is_model_image_capable(model)]
        
        # Show model selection options
        st.subheader("3. Select Bedrock Model")
        show_all_models = st.checkbox("Show all models (including those without image support)")
        
        if show_all_models:
            models_to_show = all_models
            if len(image_capable_models) < len(all_models):
                st.warning(f"Only {len(image_capable_models)} of {len(all_models)} models support image processing.")
        else:
            models_to_show = image_capable_models
            st.success(f"Showing {len(image_capable_models)} models with image support.")
        
        selected_model = st.selectbox("Choose a model:", models_to_show)
        
        # Show warning if selected model doesn't support images
        if selected_model not in image_capable_models:
            st.error(f"Warning: {selected_model} does not support image processing!")
        
        # Select output format
        st.subheader("4. Select Output Format")
        output_format = st.radio(
            "Choose output format:",
            ["JSON", "TXT", "CSV"],
            help="JSON: Structured data format\nTXT: Plain text format\nCSV: Spreadsheet format"
        )
        
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
        
        # Extract model name for display purposes
        model_name_parts = selected_model.split('.')
        if len(model_name_parts) > 1:
            model_name = model_name_parts[1].split('-')[0]  # Get the base name (claude, llama, etc.)
        else:
            model_name = selected_model
        
        # Dictionary to store all transcriptions
        transcriptions = {}
        
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
                # Create processor for this model
                processor = BedrockImageProcessor(
                    api_key="",  # Empty as we're using AWS credentials from environment
                    prompt_name=selected_prompt_name,
                    prompt_text=selected_prompt_text,
                    model=selected_model,
                    modelname=model_name
                )
                
                # Check if model supports image processing
                if not processor.supports_image_processing():
                    results.append({
                        "url": url,
                        "status": "error",
                        "message": f"Model {selected_model} does not support image processing"
                    })
                    continue
                
                # Process the image
                content, processing_data = processor.process_image(base64_image, image_name, i)
                
                # Add the transcription to the dictionary
                transcriptions[image_name] = json.loads(content) if isinstance(content, str) else content
                
                results.append({
                    "url": url,
                    "status": "success",
                    "image_name": image_name,
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
                    
                    # Display processing data
                    if result.get("processing_data"):
                        st.subheader("Processing Data")
                        for key, value in result["processing_data"].items():
                            st.write(f"{key}: {value}")
                    
                    # Display transcription content
                    image_name = result.get("image_name")
                    if image_name and image_name in transcriptions:
                        st.subheader("Transcription")
                        st.json(transcriptions[image_name])
                else:
                    st.error(f"Error: {result['message']}")
        
        # Save all transcriptions based on selected format
        if transcriptions:
            # Generate suggested volume name
            suggested_volume_name = get_volume_name(model_name)
            
            # Allow user to edit the volume name
            st.subheader("Save Transcriptions")
            st.write("Enter a name for your transcription file:")
            
            # Create two columns for the input and explanation
            col1, col2 = st.columns([3, 2])
            
            with col1:
                volume_name = st.text_input(
                    "Volume Name",
                    value=suggested_volume_name,
                    help="You can use the suggested name or enter your own"
                )
            
            with col2:
                st.info(f"Suggested name: {suggested_volume_name}")
                st.write("The file will be saved in the 'transcriptions' folder.")
            
            # Save button
            save_button = st.button("Save Transcriptions")
            
            if save_button and volume_name:
                output_file_path = ""
                file_extension = ""
                mime_type = ""
                
                if output_format == "JSON":
                    output_file_path = save_transcriptions_json(transcriptions, volume_name)
                    mime_type = "application/json"
                    file_extension = "json"
                elif output_format == "TXT":
                    output_file_path = save_transcriptions_txt(transcriptions, volume_name)
                    mime_type = "text/plain"
                    file_extension = "txt"
                elif output_format == "CSV":
                    output_file_path = save_transcriptions_csv(transcriptions, volume_name)
                    mime_type = "text/csv"
                    file_extension = "csv"
                
                # Save cost data
                cost_data_path, cost_summary = save_cost_data(
                    volume_name, 
                    selected_model, 
                    model_name, 
                    results, 
                    selected_prompt_name
                )
                
                st.success(f"All transcriptions saved to: {output_file_path}")
                st.success(f"Cost data saved to: {cost_data_path}")
                
                # Display cost summary
                st.subheader("Cost Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Cost", f"${cost_summary['costs']['total']:.4f}")
                
                with col2:
                    st.metric("Total Tokens", f"{cost_summary['tokens']['total']:,}")
                
                with col3:
                    st.metric("Processing Time", f"{cost_summary['processing_time_minutes']:.2f} min")
                
                # Add a download button for the output file
                with open(output_file_path, "r", encoding="utf-8") as f:
                    file_data = f.read()
                
                st.download_button(
                    label=f"Download Transcriptions ({output_format})",
                    data=file_data,
                    file_name=f"{volume_name}.{file_extension}",
                    mime=mime_type
                )
                
                # Add a download button for the cost data
                with open(cost_data_path, "r", encoding="utf-8") as f:
                    cost_data = f.read()
                
                st.download_button(
                    label="Download Cost Data (JSON)",
                    data=cost_data,
                    file_name=f"{volume_name}-data.json",
                    mime="application/json",
                    key="cost_data_download"  # Need a unique key for the second download button
                )

if __name__ == "__main__":
    main()