import streamlit as st
import os
import json
import csv
import re
import requests
import base64
from pathlib import Path
import time
import boto3
import datetime
import shutil
from bedrock_interface import BedrockImageProcessor
from bedrock_interface import create_image_processor
from utilities import utils

# directories
TEMP_IMAGES_DIR = "temp_images"
TRANSCRIPTIONS_DIR = "transcriptions"
RAW_RESPONSES_DIR = "raw_llm_responses"
PROMPTS_DIR = "prompts"
DATA_DIR = "data"
MODEL_INFO_DIR = "model_info"
UPLOAD_IMAGES_DIR = "images_to_upload"

# constants
TESTING_MODE = False

def initialize_variables():
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
    if 'volume_name' not in st.session_state:
        st.session_state.volume_name = ""
    if 'output_format' not in st.session_state:
        st.session_state.output_format = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = ""
    if 'selected_model_obj' not in st.session_state:
        st.session_state.selected_model_obj = None    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ""
    if 'selected_prompt_name' not in st.session_state:
        st.session_state.selected_prompt_name = ""
    if 'error_flag' not in st.session_state:
        st.session_state.error_flag = False 
    if 'proceed_option' not in st.session_state:
        st.session_state.proceed_option = None
    if 'try_failed_jobs' not in st.session_state:
        st.session_state.try_failed_jobs = False
    if 'auto_save_enabled' not in st.session_state:               
        st.session_state.auto_save_enabled = True
    if 'image_numbers' not in st.session_state:
        st.session_state.image_numbers = {}
    if 'time_start' not in st.session_state:
        st.session_state.time_start = get_timestamp()
    if 'pause_button_enabled' not in st.session_state:
        st.session_state.pause_button_enabled = False             

def compile_job(original_filename, image_idx):
    pass

def compile_successful_result():
    pass

def compile_failed_result():
    pass        

def copy_local_image(source_path, index):
    try:
        # Get the original filename
        original_filename = os.path.basename(source_path)
        # Create destination path
        dest_path = os.path.join(TEMP_IMAGES_DIR, original_filename)
        shutil.copy2(source_path, dest_path)
        return dest_path, None
    except Exception as e:
        return None, f"Error copying image: {str(e)}"

def create_costs_summary():
    cost_data_path, cost_summary = save_cost_data(
            st.session_state.volume_name, 
            st.session_state.selected_model, 
            st.session_state.model_name, 
            st.session_state.results, 
            st.session_state.selected_prompt_name
        )
    st.session_state.cost_data_path = cost_data_path
    st.session_state.cost_summary = cost_summary        

def create_directories():
    for directory in [TEMP_IMAGES_DIR, TRANSCRIPTIONS_DIR, RAW_RESPONSES_DIR, DATA_DIR, MODEL_INFO_DIR, UPLOAD_IMAGES_DIR]:
        ensure_directory_exists(directory)

# Download image from URL and save to temp folder
def download_image(url, index):
    # Get original filename from URL
    filename = url.split("/")[-1]
    file_path = os.path.join(TEMP_IMAGES_DIR, filename)
    # if image is already in temp folder, skip the download
    if os.path.exists(file_path):
        return file_path, None
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path, None
        else:
            return None, f"Failed to download image. Status code: {response.status_code}"
    except Exception as e:
        return None, f"Error downloading image: {str(e)}"

def ensure_data_is_json(data):
    return utils.parse_innermost_dict(data)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_legal_filename(filename):
    return re.sub(r'[\\/*?:]', "_", filename)        

def get_raw_llm_response(image_name):
    legal_image_name = get_legal_filename(image_name)
    raw_llm_response_path = f"raw_llm_responses/{st.session_state.volume_name}/{legal_image_name}-raw.json"
    print(f"loading {raw_llm_response_path = }")
    try:
        with open(raw_llm_response_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_saved_runs():
    saved_runs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".json") and is_incomplete_run(file):
            saved_runs.append(file)
    return saved_runs        

def get_timestamp():
    return time.strftime("%Y-%m-%d-%H%M")

def get_volume_name(model_name_short):
    return f"{model_name_short}-{st.session_state.time_start}" 

def handle_proceed_option():
    print(f"in handle_proceed_option @ {get_timestamp()}")
    proceed_option = st.session_state.get("proceed_option", "")
    st.session_state.results[-1]["proceed_option"] = proceed_option
    st.session_state.proceed_option = None
    if proceed_option == "Pause":
        st.write("Pausing....")
        return
    st.session_state.error_flag = False
    # remove failed transcripts
    failed_jobs = st.session_state.jobs_dict["failed"]
    num_failed_jobs = len(failed_jobs)
    failed_filenames = [job[1] for job in failed_jobs]
    print(f"{failed_filenames = }")
    for orig_filename in failed_filenames:
        if orig_filename in st.session_state.transcriptions:
            del st.session_state.transcriptions[orig_filename]  
    if proceed_option == "Skip Failed Jobs and Finish Remaining Jobs":
        st.write("Skipping Failed Jobs and Finishing Remaining Jobs...")
        st.session_state.jobs_dict["num_remaining_jobs"] -= num_failed_jobs
        st.session_state.jobs_dict["failed"] = []
        st.session_state.try_failed_jobs = False
        return run_jobs()
    elif proceed_option == "Substitute Blank Transcript and Finish Remaining Jobs":
        blank_transcript = utils.get_blank_transcript(st.session_state.selected_prompt_text)
        for orig_filename in failed_filenames:
            st.session_state.transcriptions[orig_filename] = blank_transcript
            st.session_state.jobs_dict["num_remaining_jobs"] -= num_failed_jobs
        st.session_state.jobs_dict["failed"] = []
        st.session_state.try_failed_jobs = False
        return run_jobs()      
    elif proceed_option == "Retry Failed and Remaining Jobs":
        st.write("Retrying Failed Jobs...")
        st.session_state.try_failed_jobs = True
        return run_jobs() 
    elif proceed_option == "Cancel All Jobs":
        st.write("Cancelling...")
        st.session_state.jobs_dict["num_remaining_jobs"] -= num_failed_jobs
        st.session_state.jobs_dict["to_process"] = []
        st.session_state.jobs_dict["failed"] = []
        st.session_state.try_failed_jobs = False
        return                                          

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8') 

def init_jobs(num_jobs):
    st.session_state.jobs_dict = {"to_process": [], "in_process": (), "failed": [], "completed": [], "incomplete": [], "msg": {}, "num_total_jobs": num_jobs, "num_remaining_jobs": num_jobs}

def is_incomplete_run(file):
    with open(os.path.join(DATA_DIR, file), "r") as f:
        data = json.load(f)
    return "incomplete" in data and data["incomplete"]

def load_job(job: dict):
    st.session_state.jobs_dict["to_process"].append(job)                      

# Load available models from vision_model_info.json
def load_models():
    try:
        with open("model_info/vision_model_info.json", "r") as f:
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

def load_saved_data(data_filename):
    with open(os.path.join(DATA_DIR, data_filename), "r") as f:
        data = json.load(f)
    st.session_state.transcriptions = data["transcriptions"]
    st.session_state.results = data["results"]
    st.session_state.volume_name = data["run_id"]
    st.session_state.selected_model = data["model"]["id"]
    st.session_state.selected_model_obj = data["model"]["id"]
    st.session_state.model_name = data["model"]["name"]
    prompt_name = data["prompt"]
    st.session_state.selected_prompt_name = prompt_name
    prompt_text = load_prompts()[prompt_name]
    st.session_state.selected_prompt_text = prompt_text
    image_data = data["images"]
    incomplete_jobs = data["incomplete"]
    return image_data

def load_saved_transcriptions(transcription_filename_no_ext):
    for file in os.listdir(TRANSCRIPTIONS_DIR):
        if file.beginswith(transcription_filename_no_ext):
            file_path = os.path.join(TRANSCRIPTIONS_DIR, file)
            st.session_state.output_file_path = file_path
            st.session_state.output_format = file_path.split(".")[-1].upper()
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()    

def load_saved_run(data_filename):
    image_data = load_saved_data(data_filename)
    transcription_filename_no_ext = data_filename.split("-data")[0] + "-transcription"
    transcriptions = load_saved_transcriptions(transcription_filename_no_ext)
    combine_
    st.session_state.cost_data_path = data["cost_data_path"]
    st.session_state.cost_summary = data["cost_summary"]
    st.session_state.file_extension = data["file_extension"]
    st.session_state.jobs_dict = data["jobs_dict"]
    st.session_state.progress = data["progress"]
    st.session_state.progress_bar.progress(st.session_state.progress)
    st.session_state.auto_save_enabled = data["auto_save_enabled"]
    st.session_state.image_numbers = data["image_numbers"]
    st.session_state.time_start = data["time_start"]
    st.session_state.pause_button_enabled = data["pause_button_enabled"]
    st.session_state.pause_button_enabled = False
    st.session_state.pause_button_placeholder.empty()
    st.session_state.pause_button_placeholder = st.empty()
    st.session_state.pause_button_placeholder.button("Pause", on_click=handle_pause_button, key="pause_button")
    st.session_state.pause_button_placeholder.empty()
    st.session_state.pause_button_placeholder = st.empty()        

def number_images(image_names):
    # {image_name: (idx, num_attempts)}
    st.session_state.image_numbers = {image_name: [idx + 1, 0] for idx, image_name in enumerate(image_names)}       

# Define the common image processing function
def process_single_image(img_path, orig_filename, item_index, source_identifier):
    print(f"Processing {orig_filename} @ {get_timestamp()}")
    processing_data, image_number, attempt_number, raw_response = None, None, None, None
    try:
        base64_image = image_to_base64(img_path)
        processor = create_image_processor(
            api_key="",  # Empty as we're using AWS credentials from environment
            prompt_name=st.session_state.selected_prompt_name,
            prompt_text=st.session_state.selected_prompt_text,
            model=st.session_state.selected_model,
            modelname=st.session_state.model_name,
            output_name=st.session_state.volume_name,
            testing=TESTING_MODE
        )
        # Process the image
        image_number, attempt_number = st.session_state.image_numbers[orig_filename]
        attempt_number += 1
        st.session_state.image_numbers[orig_filename][1] = attempt_number
        content, processing_data, raw_response = processor.process_image(base64_image, orig_filename, item_index)
        print(f"{raw_response = }")
        transcription_data = ensure_data_is_json(content)
        # Try to parse the content as JSON if it's a string
        if isinstance(transcription_data, str):
            raise Exception(f"Error processing image {orig_filename}: {transcription_data}")
        else:
            # Add the transcription to the dictionary using the original filename
            st.session_state.transcriptions[orig_filename] = transcription_data
            st.session_state.results.append({
                "imageName": source_identifier,
                "status": "success",
                "imageRef": orig_filename,
                "processing_data": processing_data,
                "image_number": image_number,
                "attempt_number": attempt_number,
                "raw_response": raw_response
            })
            st.session_state.progress = (st.session_state.jobs_dict["num_total_jobs"] - st.session_state.jobs_dict["num_remaining_jobs"]) / st.session_state.jobs_dict["num_total_jobs"]
            st.session_state.progress_bar.progress(st.session_state.progress)
            return True
    except Exception as e:
        # Create a more detailed error message
        from utilities.error_message import ErrorMessage
        error_obj = ErrorMessage.from_exception(e)
        error_msg = error_obj.get_truncated_message(1000)
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
        else:
            error_msg += "\nUnknown error: An unexpected error occurred."
        st.session_state.results.append({
            "imageName": source_identifier,
            "status": "error",
            "message": f"Error processing image: {error_msg}",
            "imageRef": orig_filename,
            "processing_data": processing_data,
            "image_number": image_number,
            "attempt_number": attempt_number,
            "raw_response": raw_response
        })
        return False

def resume_jobs(try_failed_jobs=False):
    if try_failed_jobs:
        jobs = st.session_state.jobs_dict
        while jobs["failed"]:
            job_to_retry = jobs["failed"].pop(-1)
            jobs["to_process"].insert(0, job_to_retry)
    return run_jobs()

def run_jobs():
    print(f"in run_jobs @ {get_timestamp()}")
    jobs = st.session_state.jobs_dict
    if jobs["failed"] and st.session_state.try_failed_jobs:
        while jobs["failed"]:
            job_to_retry = jobs["failed"].pop(-1)
            jobs["to_process"].insert(0, job_to_retry)
        st.session_state.try_failed_jobs = False    
    while jobs["to_process"]:
        jobs["in_process"] = jobs["to_process"].pop(0)
        img_path, orig_filename, item_index, source_identifier = jobs["in_process"]
        is_successful_job = process_single_image(img_path, orig_filename, item_index, source_identifier)
        print(f"run_jobs: {is_successful_job = }, {jobs['in_process'] = }")
        if not is_successful_job:
            jobs["failed"].append(jobs["in_process"])
            jobs["incomplete"].append(orig_filename)
            jobs["in_process"] = ()
            jobs["msg"] = st.session_state.results[-1]
            st.session_state.error_flag = True
            if st.session_state.auto_save_enabled:
                save_transcriptions_callback()
            return False
        jobs["completed"].append(orig_filename)
        if orig_filename in jobs["incomplete"]:
            jobs["incomplete"].remove(orig_filename)
        jobs["msg"] = st.session_state.results[-1]
        jobs["in_process"] = ()        
        jobs["num_remaining_jobs"] -= 1
        if st.session_state.auto_save_enabled:
                save_transcriptions_callback()    
    st.session_state.error_flag = False          
    return True

def save_cost_data(volume_name, model_id, model_name, results, prompt_name):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{DATA_DIR}/{volume_name}-data.json"
    # Calculate total tokens and costs
    total_input_tokens = 0
    total_output_tokens = 0
    total_input_cost = 0.0
    total_output_cost = 0.0
    total_time = 0.0
    image_costs = {}
    
    # Extract processing data from successful results
    for result in st.session_state.results:
        if result["status"] == "success" and "processing_data" in result:
            data = result["processing_data"]
            image_costs[result["imageName"]] = data
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
        "images_processed": len([r for r in st.session_state.results if r["status"] == "success"]),
        "images_failed": len([r for r in st.session_state.results if r["status"] == "error"]),
        "completed_jobs": st.session_state.jobs_dict["completed"],
        "incomplete_jobs": st.session_state.jobs_dict["incomplete"],
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
    cost_data["images"] = image_costs
    # Save the cost data to the JSON file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved cost data to {filename}")
    except Exception as e:
        print(f"Error saving cost data: {str(e)}")
        st.error(f"Error saving cost data: {str(e)}")
    return filename, cost_data

def save_transcriptions_callback():
    output_file_path = ""
    try:
        if st.session_state.output_format == "JSON":
            output_file_path = save_transcriptions_json(st.session_state.transcriptions, st.session_state.volume_name)
        elif st.session_state.output_format == "TXT":
            output_file_path = save_transcriptions_txt(st.session_state.transcriptions, st.session_state.volume_name)
        elif st.session_state.output_format == "CSV":
            output_file_path = save_transcriptions_csv(st.session_state.transcriptions, st.session_state.volume_name)
        print(f"File saved successfully: {output_file_path}")
        st.session_state.output_file_path = output_file_path
        st.session_state.save_completed = True
        # Don't rerun the app, just set a flag to show success message
        st.session_state.show_save_success = True
    except Exception as e:
        print(f"Error in save_transcriptions_callback: {str(e)}")
        st.error(f"Error saving files: {str(e)}")
        st.session_state.show_save_error = str(e)
  
def save_transcriptions_csv(transcriptions, volume_name):
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.csv"
    try:
        fieldnames = ["imageName"]  # Start with image_name as the first field
        for image_name, data in transcriptions.items():
            if isinstance(data, dict):
                for key in data.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            else:
                # If transcription is not a dict, add it as a single field
                if "transcription" not in fieldnames:
                    fieldnames.append("transcription")
        # Write the CSV file
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for image_name, data in transcriptions.items():
                for fieldname, val in data.items():
                    data[fieldname] = val.replace('\n', ' ').replace('\r', ' ')
                data = {"imageName": image_name} | data
                writer.writerow(data)
        print(f"Successfully saved CSV transcriptions to {filename}")
    except Exception as e:
        print(f"Error saving CSV transcriptions: {str(e)}")
        st.error(f"Error saving CSV transcriptions: {str(e)}")
    return filename

def save_transcriptions_json(transcriptions, volume_name):
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)
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

def save_transcriptions_txt(transcriptions, volume_name):
    filename = f"{TRANSCRIPTIONS_DIR}/{volume_name}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for i, (image_name, data) in enumerate(transcriptions.items()):
                if i > 0:
                    f.write("\n\n")
                # Add header separator
                f.write("=" * 50 + "\n")
                f.write(f"imageName: {image_name}\n\n")
                transcription_data = {}
                if isinstance(data, dict):
                    transcription_data = data
                else:
                    transcription_data = {"transcription": data}
                for key, value in transcription_data.items():
                    formatted_value = str(value).strip()
                    f.write(f"{key}: {formatted_value}\n")
                # Add footer separator
                f.write("\n" + "=" * 50)
        print(f"Successfully saved TXT transcriptions to {filename}")
    except Exception as e:
        print(f"Error saving TXT transcriptions: {str(e)}")
        st.error(f"Error saving TXT transcriptions: {str(e)}")
    return filename

def set_start_time():
    if "start_time" not in st.session_state:
        st.session_state.start_time_str = get_timestamp()    


def main():
    st.set_page_config(
    page_title="Bedrock Image Transcription App",
    page_icon="🖼️",
    layout="wide"
    )
    st.title("Bedrock Image Transcription App")
    create_directories()
    st.session_state.show_save_success = False
    if st.sidebar.button("Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("App reset successfully!")
        st.rerun()
    set_start_time()    
    st.session_state.auto_save_enabled = st.toggle(label="Auto Save", key="auto_save_option", value=True)    
    all_models = load_models()
    prompts = load_prompts()
    with st.sidebar:
        st.session_state.task_option = "New Run"#= st.radio("Choose Operation", ["New Run", "Complete Saved Run"], index=0, key="selected_task")
        ######### NEW RUN SIDEBAR  ############
        # 1. Model
        # 2. Prompt
        # 3. Input Method (url or local)
        # 4. Name for Saving Output
        # 5. Format for Saving Output
        # 6. Process Button
        if st.session_state.task_option == "New Run":
            st.header("Configuration")
            # 1. Begin Model Selection
            st.subheader("1. Select Bedrock Model")
            model_filter = st.radio(
                "Filter models by:",
                ["Models that passed image test", "All image-capable models", "All models"],
                index=0
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
            #if mistral_count > 0:
            #    st.info(f"Note: {mistral_count} Mistral models have been excluded to prevent terminal flooding issues.")
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
            # End Model Selection
            # 2. Begin Prompt Selection
            st.subheader("2. Select a Prompt")
            selected_prompt_name = st.selectbox("Choose a prompt:", list(prompts.keys()))
            selected_prompt_text = prompts[selected_prompt_name]
            with st.expander("View Selected Prompt"):         # Display selected prompt
                st.text_area("Prompt Content", selected_prompt_text, height=200, disabled=True)
            # End Prompt Selection
            # 3. Begin Input Method Selection
            st.subheader("3. Select Input Method")
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
            # End Input Method Selection
            # 4. Begin Naming Output
            st.subheader("4. Name Output")
            if selected_model:
                model_name = selected_model.split(':')[0]
                suggested_volume_name = get_volume_name(model_name)
                suggested_volume_name = get_legal_filename(suggested_volume_name)
                # Allow user to edit the volume name
                st.warning("Suggested Output Name:")
                st.info(suggested_volume_name)
                volume_name = st.text_input(
                    "You May Edit the Name Below. 'Enter' to Accept Changes",
                    value=suggested_volume_name,
                    help="You can use the suggested name or enter your own",
                    key="volume_name_input"
                )
                st.session_state.volume_name = volume_name
                st.write("Look For") 
                st.success(f"'transcriptions/{volume_name}-transcription'")
                st.write("  And")
                st.success(f"'data/{volume_name}-data'")

            else:
                st.warning("Please select a model to generate a suggested name.")
            # End Naming Output
            # 5. Begin Selection of Output File Format
            st.subheader("5. Select File Format for Saving")
            st.session_state.output_format = st.radio(
                "Choose output format:",
                ["JSON", "CSV", "TXT"],
                help="JSON: Structured data format\nCSV: Spreadsheet format\nTEXT: a single plain text file"
            )
            # End Selection of Output File Format
            # 6. Process button - disable if no input is provided
            process_button_disabled = (input_method == "Upload URLs File" and not uploaded_file) or \
                                    (input_method == "Select Local Images" and not selected_local_images)
            #process_button_disabled = False
            process_button_clicked = st.button("Process Images", type="primary", disabled=process_button_disabled)
            # End 6.
            # persistance
            st.session_state.selected_model = selected_model
            st.session_state.model_name = model_name
            st.session_state.selected_model_obj = selected_model_obj
            st.session_state.selected_prompt_name = selected_prompt_name
            st.session_state.selected_prompt_text = selected_prompt_text
        ##### End New Run Sidebar
        else:
            # Complete Saved Run Sidebar
            st.header("Complete Saved Run")
            saved_runs = get_saved_runs()
            if not saved_runs:
                st.warning("No saved runs found.")
                return
            selected_run = st.selectbox("Select a saved run:", saved_runs)
            if st.button("Load Run"):
                load_saved_run(selected_run)
                process_button_clicked = True
                st.rerun()
        
    # Main content area
    ## begin processing images
    st.session_state.progress_bar = st.progress(st.session_state.get("progress", 0))
    status_text = st.empty()
    if process_button_clicked:
        st.success("Processing started...")
        has_valid_input = False
        urls = []
        local_image_paths = []
        # create list of urls
        if input_method == "Upload URLs File" and uploaded_file is not None:
            urls = uploaded_file.getvalue().decode("utf-8").splitlines()
            urls = [url.strip() for url in urls if url.strip()]
            number_images(urls)
            if not urls:
                st.error("No URLs found in the uploaded file.")
                return
            has_valid_input = True
            st.info(f"Processing {len(urls)} images from URLs...")
        # create list of local image paths
        elif input_method == "Select Local Images" and selected_local_images:
            number_images(selected_local_images)
            local_image_paths = [os.path.join(UPLOAD_IMAGES_DIR, img) for img in selected_local_images]
            has_valid_input = True
            st.info(f"Processing {len(local_image_paths)} local images...")
        if not has_valid_input:
            st.error("Please provide input images (either upload a URL file or select local images).")
            return
        # Start Setting Up Jobs
        st.session_state.total_items = len(urls) + len(local_image_paths)
        init_jobs(st.session_state.total_items)      
        # Begin Loading URL images
        for i, url in enumerate(urls):
            # Get the original filename from the URL
            original_filename = url #url.split("/")[-1]
            image_path, error = download_image(url, i)
            if error:
                st.session_state.results.append({"imageName": url, "status": "error", "message": error})
                continue
            job = (image_path, original_filename, i, url)    
            load_job(job)
        # End Loading URL Images    
        # Begin Loading Local Images
        for i, local_path in enumerate(local_image_paths):
            filename = os.path.basename(local_path)
            # Copy image to temp folder
            image_path, error = copy_local_image(local_path, i)
            if error:
                st.session_state.results.append({"imageName": filename, "status": "error", "message": error})
                continue
            job = (image_path, filename, len(urls) + i, filename)
            load_job(job)
        # End Loading Local Images
        # Begin Processing Jobs
        st.session_state.progress = (st.session_state.jobs_dict["num_total_jobs"] - st.session_state.jobs_dict["num_remaining_jobs"]) / st.session_state.jobs_dict["num_total_jobs"]
        st.session_state.progress_bar.progress(st.session_state.get("progress", 0))
        st.session_state.pause_button_enabled = False#st.toggle(label="Pause", key="pause_button", value=False)
        if st.session_state.jobs_dict["to_process"] and not st.session_state.pause_button_enabled:
            is_succcess = run_jobs()
    if st.session_state.error_flag:
        msg = st.session_state.results[-1]["message"]
        print(f"Error indicated @ {get_timestamp()}")
        print(f"{msg = }")
        st.error("Error!!!")
        st.error(msg)
        proceed_option = st.radio("How to Proceeed?:", ["Pause", "Retry Failed and Remaining Jobs", "Substitute Blank Transcript and Finish Remaining Jobs", "Skip Failed Jobs and Finish Remaining Jobs", "Cancel All Jobs"], index=None, key="proceed_option", on_change=handle_proceed_option)
    if st.session_state.results:
        # Begin Display Results
        st.header("Results")
        success_count = sum(1 for r in st.session_state.results if r["status"] == "success")
        error_count = sum(1 for r in st.session_state.results if r["status"] == "error")
        st.write(f"Successfully processed: {success_count} images")
        st.write(f"Errors: {error_count} images")
        # Show detailed results
        for i, result in enumerate(st.session_state.results):
            # Get the filename for display (could be URL or local filename)
            display_name = result['imageName']
            image_number, __ = st.session_state.image_numbers.get(display_name, (None, None))
            attempt_number = result.get("attempt_number", 1)
            if '/' in display_name:
                display_name = display_name.split("/")[-1]
            st.session_state[f"expander_{display_name}"] = st.expander(f"Image {image_number}, Attempt {attempt_number}: {display_name}")
            with st.session_state[f"expander_{display_name}"]:
                st.write(f"Image {image_number}, Attempt {attempt_number}: {display_name}")
                if result["status"] == "success":
                    st.success("Successfully processed")
                    # Display processing data
                    if result.get("processing_data"):
                        st.subheader("Processing Data")
                        for key, value in result["processing_data"].items():
                            st.write(f"{key}: {value}")
                    # Display transcription content
                    image_name = result.get("imageName")
                    if image_name and image_name in st.session_state.transcriptions:
                        st.subheader("Transcription")
                        display_data = st.session_state.transcriptions[image_name]
                        if isinstance(display_data, dict):
                            st.json(display_data)
                        else:    
                            try:
                                # Try to parse the JSON string for display
                                parsed_json = json.loads(display_data)
                                st.json(parsed_json)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, display as is
                                st.json(display_data)
                        st.caption(f"Original filename: {image_name}")
                else:
                    st.error(f"Error: {result['message']}")
                    selected_proceed_option = result.get("proceed_option", None)
                    st.write(f"Response to Error: {selected_proceed_option}")
                    raw_llm_response = result.get("raw_response")
                    if not raw_llm_response:
                        st.warning("No raw LLM response available.")    
                    elif st.toggle("Show Raw LLM Response (re-expand box after toggling)", key=f"{display_name}: {attempt_number}"):
                        st.write(raw_llm_response)
                        #st.json(raw_llm_response)
                    
        # End Display Results
        # Begin Save Results
    if st.session_state.transcriptions:
        create_costs_summary()
        st.subheader("Save Transcriptions")
        save_btn = st.button("Save Transcriptions", key="save_button", on_click=save_transcriptions_callback)
        # Display cost summary
        st.subheader("Cost Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cost", f"${st.session_state.cost_summary['costs']['total']:.4f}")
        with col2:
            st.metric("Total Tokens", f"{st.session_state.cost_summary['tokens']['total']:,}")
        with col3:
            st.metric("Processing Time", f"{st.session_state.cost_summary['processing_time_minutes']:.2f} min")
        # After displaying the results, check if files were saved
    if "save_completed" in st.session_state and st.session_state.save_completed:
        if "show_save_success" in st.session_state and st.session_state.show_save_success:
            st.success(f"All transcriptions saved to: {st.session_state.output_file_path}")
            st.success(f"Cost data saved to: {st.session_state.cost_data_path}")
            
            
if __name__ == "__main__":
    initialize_variables()
    main()