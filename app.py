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
from dotenv import load_dotenv
from bedrock_interface import BedrockImageProcessor
from bedrock_interface import create_image_processor
from input_output_manager import InputOutputManager
from utilities import utils
from utilities.adjust_costs import main as adjust_costs
from utilities.error_message import ErrorMessage

load_dotenv()
TESTING_MODE = os.getenv("TESTING_MODE", "False").lower() == "true"
RUN_PREFIX = "test-" if TESTING_MODE else ""

# directories
TEMP_IMAGES_DIR = "temp_images"
TRANSCRIPTIONS_DIR = "transcriptions"
RAW_RESPONSES_DIR = "raw_llm_responses"
PROMPTS_DIR = "prompts"
DATA_DIR = "data"
MODEL_INFO_DIR = "model_info"
UPLOAD_IMAGES_DIR = "images_to_upload"
RECOVERY_DIR = "recovery"

def initialize_variables():
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'save_completed' not in st.session_state:
        st.session_state.save_completed = False
    if 'cost_data_path' not in st.session_state:
        st.session_state.cost_data_path = ""
    if 'cost_summary' not in st.session_state:
        st.session_state.cost_summary = {}
    if 'volume_name' not in st.session_state:
        st.session_state.volume_name = ""
    if 'output_format' not in st.session_state:
        st.session_state.output_format = "" 
    if 'error_flag' not in st.session_state:
        st.session_state.error_flag = False 
    if 'proceed_option' not in st.session_state:
        st.session_state.proceed_option = None
    if 'try_failed_jobs' not in st.session_state:
        st.session_state.try_failed_jobs = False
    if 'run_numbering' not in st.session_state:
        st.session_state.run_numbering = {}
    if 'time_start' not in st.session_state:
        st.session_state.time_start = get_timestamp()
    if 'pause_button_enabled' not in st.session_state:
        st.session_state.pause_button_enabled = False
    if 'output_files' not in st.session_state:
        st.session_state.output_files = []
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 10000
    if "ignore_throttling_errors" not in st.session_state:
        st.session_state.ignore_throttling_errors = False
    if "io_manager" not in st.session_state:
        st.session_state.io_manager = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = ""
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = ""    
    if "selected_prompt_name" not in st.session_state:
        st.session_state.selected_prompt_name = ""
    if "selected_prompt_text" not in st.session_state:
        st.session_state.selected_prompt_text = ""    
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "selected_local_images" not in st.session_state:
        st.session_state.selected_local_images = []
    if "task_option" not in st.session_state:
        st.session_state.task_option = ""
    if "process_button_clicked" not in st.session_state:
        st.session_state.process_button_clicked = False                                                         

def clear_variables():
    st.session_state.results = []
    st.session_state.save_completed = False
    st.session_state.cost_data_path = ""
    st.session_state.cost_summary = {}
    st.session_state.volume_name = ""
    st.session_state.output_format = ""
    st.session_state.error_flag = False 
    st.session_state.proceed_option = None
    st.session_state.try_failed_jobs = False
    st.session_state.run_numbering = {}
    st.session_state.time_start = get_timestamp()
    st.session_state.pause_button_enabled = False 
    st.session_state.output_files = []
    st.session_state.chunk_size = 1000
    st.session_state.ignore_throttling_errors = False
    st.session_state.io_manager = None
    st.session_state.task_option = ""
    st.session_state.process_button_clicked = False
    st.session_state.uploaded_file = None
    st.session_state.selected_local_images = []
    st.session_state.input_method = ""
    st.session_state.selected_model = ""
    st.session_state.selected_model_name = ""
    st.session_state.model_name = ""
    st.session_state.selected_prompt_name = ""
    st.session_state.selected_prompt_text = "" 
    # Explicitly reset the radio button key
    if "selected_task" in st.session_state:
        del st.session_state["selected_task"]

def address_error():
    msg = st.session_state.results[-1]["message"]
    print(f"Error indicated @ {get_timestamp()}")
    print(f"{msg = }")
    st.error("Error!!!")
    st.error(msg)
    proceed_options = get_proceed_options(msg)
    if "Substitute Blank Transcript for ALL THROTTLING ERRORS" in proceed_options and st.session_state.ignore_throttling_errors:
        st.session_state.proceed_option = "Substitute Blank Transcript for ALL THROTTLING ERRORS"
        handle_proceed_option()
        st.rerun()
    else:
        st.radio("How to Proceeed?:", proceed_options, index=None, key="proceed_option", on_change=handle_proceed_option)        

def begin_processing():
    st.session_state.total_items = len(st.session_state.run_numbering)
    init_jobs(len(st.session_state.run_numbering))
    load_jobs()
    st.session_state.progress = (st.session_state.jobs_dict["num_total_jobs"] - st.session_state.jobs_dict["num_remaining_jobs"]) / st.session_state.jobs_dict.get("num_total_jobs", 1)
    st.session_state.progress_bar.progress(max(st.session_state.progress, 0))
    st.session_state.pause_button_enabled = False #st.toggle(label="Pause", key="pause_button", value=False)
    if st.session_state.jobs_dict["to_process"] and not st.session_state.pause_button_enabled:
        run_jobs()        

def configure_inputs():
    #with st.session_state.configuration_container:
        st.header("Configuration")
        st.subheader("1. Select Bedrock Model")
        select_model()
        st.subheader("2. Select a Prompt")
        select_prompt()
        st.subheader("3. Select Input Method")
        select_input_method()
        st.subheader("4. Name Output File")
        if st.session_state.selected_model and (st.session_state.uploaded_file or st.session_state.selected_local_images):
            name_output_file()
        st.subheader("5. Select File Format for Saving")
        if st.session_state.uploaded_file or st.session_state.selected_local_images:
            select_output_format()                                                                  
                               
def create_costs_summary():
    cost_data_path, cost_summary = save_cost_data()
    st.session_state.cost_data_path = cost_data_path
    st.session_state.cost_summary = cost_summary        

def create_directories():
    for directory in [TEMP_IMAGES_DIR, TRANSCRIPTIONS_DIR, RAW_RESPONSES_DIR, DATA_DIR, MODEL_INFO_DIR, UPLOAD_IMAGES_DIR, RECOVERY_DIR]:
        ensure_directory_exists(directory)

def display_costs_summary():
    with st.session_state.costs_container:
        st.subheader("Cost Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Input Tokens", f"{st.session_state.cost_summary['tokens']['input']:,}")
            st.metric("Total Output Tokens", f"{st.session_state.cost_summary['tokens']['output']:,}")
            st.metric("Num Processed Successfully", f"{st.session_state.cost_summary['images_processed']:,}")
        with col2:
            st.metric("Input Cost Per Mil", f"{st.session_state.cost_summary['costs']['input_cost_per_mil']:.2f}")
            st.metric("Output Cost Per Mil", f"{st.session_state.cost_summary['costs']['output_cost_per_mil']:.2f}")
            st.metric("Processing Time", f"{st.session_state.cost_summary['processing_time_minutes']:.2f} min")
        with col3:
            st.metric("Total Input Cost", f"${st.session_state.cost_summary['costs']['input']:.4f}")
            st.metric("Total Output Cost", f"${st.session_state.cost_summary['costs']['output']:.4f}")
            st.metric("Total Overall Cost", f"${st.session_state.cost_summary['costs']['total']:.4f}")        

def display_file_saving_success():
    with st.session_state.file_saving_success_container:
        if "show_save_error" in st.session_state and st.session_state.show_save_error:
            st.error(f"Error saving files: {st.session_state.save_error_message}")
            st.session_state.show_save_error = False
        elif "show_save_success" in st.session_state and st.session_state.show_save_success:
            for filename in st.session_state.output_files:
                st.success(f"File saved successfully: {filename}")
            st.success(f"Cost data saved to: {st.session_state.cost_data_path}")

def display_model_details(selected_model_obj):
    with st.expander("Model Details"):
        st.json({
            "Model ID": selected_model_obj.get("modelId", ""),
            "Provider": selected_model_obj.get("provider", ""),
            "Pricing": selected_model_obj.get("pricing", ""),
            "Pricing Verified": selected_model_obj.get("pricing_verified", False),
            "Supports Image": selected_model_obj.get("supports_image", False),
            "Image Test Success": selected_model_obj.get("image_test_success", False),
            "Uses Inference Profile": selected_model_obj.get("use_inference_profile", False),
            "On-Demand Supported": selected_model_obj.get("on_demand_supported", False)
        })           

def display_results():
    with st.session_state.results_container:
        st.session_state.display_images = st.toggle("Display Images", value=True)
        for result in st.session_state.results:
            image_info, attempt_number, processing_data = result["image_info"], result["attempt_number"], result["processing_data"]
            image_name, image_number, image_path, transcription, raw_llm_response = image_info.image_name, image_info.image_number, image_info.image_path, image_info.transcription, image_info.raw_llm_response[attempt_number][0]
            display_name = image_name.split("/")[-1]
            # use a unique expander name in case of multiple attempts
            st.session_state[f"expander_{display_name}"] = st.expander(f"Image {image_number}, {display_name}, Attempt {attempt_number}: {result['status'].upper()}")
            with st.session_state[f"expander_{display_name}"]:
                if result["status"] == "success":
                    display_successful_result_details(display_name, image_name, image_path, transcription, processing_data)
                else:
                    display_unsuccessful_results_details(display_name, image_name, image_path, result, raw_llm_response)

def display_selected_prompt_text(selected_prompt_text):
    with st.expander("View Selected Prompt"):
        st.text_area("Prompt Content", selected_prompt_text, height=200, disabled=True)

def display_success_counts():
    with st.session_state.success_counts_container:
        success_count = sum(1 for r in st.session_state.results if r["status"] == "success")
        error_count = sum(1 for r in st.session_state.results if r["status"] == "error")
        st.write(f"Successfully processed: {success_count} images")
        st.write(f"Errors: {error_count} images")

def display_successful_result_details(display_name, image_name, image_path, transcription, processing_data):
    st.success("Successfully processed")
    # Display the image if path exists
    if st.session_state.display_images and image_path and os.path.exists(image_path):
        st.image(image_path, caption=f"Image: {display_name}")
    if transcription:
        st.subheader("Transcription")
        if isinstance(transcription, dict):
            st.json(transcription)
        else:    
            try:
                # Try to parse the JSON string for display
                parsed_json = json.loads(transcription)
                st.json(parsed_json)
            except json.JSONDecodeError:
                # If it's not valid JSON, display as is
                st.json(transcription)
        st.caption(f"Original filename: {image_name}")
    if processing_data:    
        st.subheader("Processing Data")
        for key, value in processing_data.items():
            st.write(f"{key}: {value}") 

def display_unsuccessful_results_details(display_name, image_name, image_path, result, raw_llm_response):
    st.error(f"Error: {result['message']}")
    # Display the image if path exists
    if st.session_state.display_images and image_path and os.path.exists(image_path):
        st.image(image_path, caption=f"Image: {display_name}")
    selected_proceed_option = result.get("proceed_option", None)
    st.write(f"Response to Error: {selected_proceed_option}")
    if not raw_llm_response:
        st.warning("No raw LLM response available.")    
    else:# st.toggle("Show Raw LLM Response (may need to re-expand box after toggling)", key=f"Raw Data {image_number}, Attempt {attempt_number}: {display_name}"):
        st.write(raw_llm_response)                               

def ensure_data_is_json(data):
    d = utils.parse_innermost_dict(data)
    return d if type(d)==dict else {"transcription": d}

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_io_error_message():
    msg = "\n".join(st.session_state.io_manager.msg["error"])
    st.session_state.io_manager.msg["error"] = []
    st.session_state.io_manager.error_flag = False
    return msg or "no error messages"

def get_io_manager(run_name, model, model_name, prompt_name, prompt_text, output_format):
    return InputOutputManager(run_name, model, model_name, prompt_name, prompt_text, output_format)       

def get_legal_filename(filename):
    return re.sub(r'[\\/*?: ]', "_", filename)

def get_max_chunk_size(uploaded_file, selected_local_images):
    if uploaded_file:   
        urls = uploaded_file.getvalue().decode("utf-8").splitlines()
        return len(urls)  
    return len(selected_local_images) or st.session_state.chunk_size

def get_more_error_details(error_msg, e):
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
    return error_msg       

def get_proceed_options(msg):
    proceed_options = ["Pause", "Retry Failed and Remaining Jobs", "Substitute Blank Transcript and Finish Remaining Jobs", "Skip Failed Jobs and Finish Remaining Jobs", "Cancel All Jobs"]
    if "throttling" in msg.lower():
        proceed_options.append("Substitute Blank Transcript for ALL THROTTLING ERRORS")
    return proceed_options          

def get_raw_llm_response(image_name):
    legal_image_name = get_legal_filename(image_name)
    raw_llm_response_path = f"raw_llm_responses/{st.session_state.volume_name}/{legal_image_name}-raw.json"
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

def get_task_options():
    return  ["New Run", "Complete Saved Run", "Reset App"]    

def get_timestamp():
    return time.strftime("%Y-%m-%d-%H%M")

def get_volume_name(model_name_short):
    return f"{model_name_short}-{st.session_state.time_start}" 

def handle_proceed_option():
    print(f"in handle_proceed_option @ {get_timestamp()}")
    proceed_option = st.session_state.get("proceed_option", "")
    st.session_state.results[-1]["proceed_option"] = proceed_option
    st.session_state.proceed_option = None
    st.session_state.ignore_throttling_errors = proceed_option == "Substitute Blank Transcript for ALL THROTTLING ERRORS"
    if proceed_option == "Pause":
        st.write("Pausing....")  
        return
    jobs = st.session_state.jobs_dict    
    sanitize_transcriptions(jobs["failed"])    
    st.session_state.error_flag = False
    if proceed_option == "Retry Failed and Remaining Jobs":
        st.write("Retrying Failed Jobs...")
        st.session_state.try_failed_jobs = True
        return run_jobs()
    failed_jobs = jobs["failed"]    
    jobs["num_remaining_jobs"] += len(jobs["failed"])
    jobs["failed"] = []
    st.session_state.try_failed_jobs = False  
    if proceed_option == "Cancel All Jobs":
        st.write("Cancelling...")
        st.session_state.jobs_dict["to_process"] = []
        return  
    elif proceed_option == "Skip Failed Jobs and Finish Remaining Jobs":
        st.write("Skipping Failed Jobs and Finishing Remaining Jobs...")
    elif proceed_option == "Substitute Blank Transcript for ALL THROTTLING ERRORS" or proceed_option == "Substitute Blank Transcript and Finish Remaining Jobs":
        st.write("Substituting Blank Transcript...")
        print("Substituting Blank Transcript...")
        blank_transcript = utils.get_blank_transcript(st.session_state.io_manager.prompt_text)
        for image_info in failed_jobs:
            image_info.set_transcription(blank_transcript, st.session_state.io_manager.fieldnames)
            save_transcription(image_info.image_number)
    return run_jobs()        
 
def init_jobs(num_jobs):
    st.session_state.jobs_dict = {"to_process": [], "in_process": (), "failed": [], "completed": [], "incomplete": [], "msg": {}, "num_total_jobs": num_jobs, "num_remaining_jobs": num_jobs}

def is_incomplete_run(file):
    with open(os.path.join(DATA_DIR, file), "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            error_msg = ErrorMessage(str(e))
            data = f.read()
            #print(f"Error decoding JSON: {data = }")
            #print(f"Error message: {error_msg}")
            return False
    return "incomplete_jobs" in data and data["incomplete_jobs"]

def load_failed_jobs(jobs):
    while jobs["failed"]:
        job_to_retry = jobs["failed"].pop(-1)
        jobs["to_process"].insert(0, job_to_retry)
    st.session_state.try_failed_jobs = False
    return jobs       

def load_job(job: dict):
    st.session_state.jobs_dict["to_process"].append(job)
    st.session_state.jobs_dict["incomplete"].append(job.image_name)    

def load_jobs():
    run_numbering = st.session_state.io_manager.get_run_numbering()
    jobs = [image_info for image_info in run_numbering.values() if not image_info.has_completed_transcription]
    st.session_state.jobs_dict["num_total_jobs"] = len(jobs)
    st.session_state.jobs_dict["num_remaining_jobs"] = len(jobs)
    for job in jobs:
        load_job(job)                      

# Load available models from vision_model_info.json
def load_models():
    try:
        with open("model_info/vision_model_info.json", "r") as f:
            models = json.load(f)
        # Add a display name for each model
        successful_models = [model for model in models if "image_test_success" in model and model["image_test_success"]]
        #print(f"{successful_models = }")
        for model in successful_models :
            model_id = model.get("modelId", "")
            model_name = model.get("modelName", "")
            provider = model.get("provider", "")
            model["display_name"] = f"{model_name} ({provider})"
        return successful_models
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
    st.session_state.time_start = get_timestamp()
    selected_model = data["model"]
    model_name = data["model_name"]
    prompt_name = data["prompt_name"]
    prompt_text = load_prompts()[prompt_name]
    missing_transcriptions = data["incomplete_jobs"]
    st.session_state.volume_name = data["run_id"]
    output_format = data["output_format"].split(".")[-1].upper()
    st.session_state.chunk_size = data["chunk_size"]
    st.session_state.io_manager = InputOutputManager(st.session_state.volume_name, selected_model, model_name, prompt_name, prompt_text, output_format)
    st.session_state.run_numbering = st.session_state.io_manager.load_run_numbering(data["run_numbering"], st.session_state.chunk_size)    
    return missing_transcriptions       

def load_saved_run(data_filename):
    missing_transcriptions = load_saved_data(data_filename)
    return missing_transcriptions

def move_to_completed_list(jobs, image_info):
    jobs["completed"].append(image_info.image_name)
    if image_info.image_name in jobs["incomplete"]:
        jobs["incomplete"].remove(image_info.image_name)
    jobs["msg"] = st.session_state.results[-1]
    jobs["in_process"] = ()        
    jobs["num_remaining_jobs"] -= 1      

def move_to_failed_list(jobs, image_info):
    jobs["failed"].append(image_info)
    if image_info.image_name not in jobs["incomplete"]:
        jobs["incomplete"].append(image_info.image_name)
    jobs["in_process"] = ()
    jobs["msg"] = st.session_state.results[-1] 

def name_output_file():
    st.session_state.model_name = st.session_state.selected_model.split(':')[0]
    suggested_volume_name = get_volume_name(st.session_state.model_name)
    suggested_volume_name = get_legal_filename(suggested_volume_name)
    # Allow user to edit the volume name
    volume_name = st.text_input(
        "You May Edit the Name Below. 'Enter' to Accept Changes",
        value=suggested_volume_name,
        help="You can use the suggested name or enter your own",
        key="volume_name_input"
    )
    st.session_state.volume_name = get_legal_filename(volume_name)
    st.write("Look For the Transcription Folder:") 
    st.success(f"'transcriptions/{st.session_state.volume_name}-transcription'")
    st.write("  And the Data File:")
    st.success(f"'data/{st.session_state.volume_name}-data'")

def pre_process_inputs():
    has_valid_input = False
    urls = []
    local_image_paths = []
    # create list of urls
    if st.session_state.input_method == "Upload URLs File" and st.session_state.uploaded_file:
        urls = st.session_state.uploaded_file.getvalue().decode("utf-8").splitlines()
        urls = [url.strip() for url in urls if url.strip()]
        if not urls:
            st.error("No URLs found in the uploaded file.")
            return
        st.info(f"Processing {len(urls)} images from URLs...")    
        st.session_state.run_numbering = st.session_state.io_manager.set_run_numbering(images_to_process=urls, use_urls=True, chunk_size=st.session_state.chunk_size)
    # create list of local image paths
    elif st.session_state.input_method == "Select Local Images" and st.session_state.selected_local_images:
        st.session_state.run_numbering = st.session_state.io_manager.set_run_numbering(images_to_process=st.session_state.selected_local_images, use_urls=False, chunk_size=st.session_state.chunk_size)
        st.info(f"Processing {len(st.session_state.selected_local_images)} local images...")   
    st.session_state.total_items = len(st.session_state.run_numbering)    
    io_error_msg = get_io_error_message()
    if io_error_msg == "no error messages":
        has_valid_input = True
    else:
        st.error(io_error_msg)
        st.info(f"{st.session_state.total_items} available to process")
        go_ahead_and_process_option = st.radio("Proceed?", ["Yes", "No (Cancel)"])
        if go_ahead_and_process_option == "No (Cancel)":
            has_valid_input = False
        if go_ahead_and_process_option == "Yes":
            has_valid_input = True
            st.info(f"Processing {number_images} images...")    
    if not has_valid_input:
        st.error("Please provide input images (either upload a URL file or select local images).")               

# Define the common image processing function
def process_single_image(image_info):
    image_name, image_number, base64_image = image_info.image_name, image_info.image_number, image_info.base64_image
    print(f"Processing {image_name} @ {get_timestamp()}")
    processing_data, raw_response = None, None 
    try:
        processor = st.session_state.io_manager.processor
        image_info.increment_number_attempts()
        content, processing_data, raw_response = processor.process_image(base64_image, image_name, image_number)
        if "error" in processing_data:
            raise Exception(processing_data["error"])
        transcription_data = ensure_data_is_json(content)
        if isinstance(transcription_data, str):
            raise Exception(f"Error processing image {image_name}: {transcription_data}")
        else:
            image_info.set_transcription(transcription_data, st.session_state.io_manager.fieldnames)
            image_info.set_raw_llm_response(raw_llm_response=raw_response, is_associated_with_error=False)
            image_info.add_processing_data_to_image_data(processing_data)
            st.session_state.results.append({"image_info": image_info, "attempt_number": image_info.attempt_number, "status": "success", "processing_data": processing_data})
            st.session_state.progress = (st.session_state.jobs_dict["num_total_jobs"] - st.session_state.jobs_dict["num_remaining_jobs"]) / st.session_state.jobs_dict["num_total_jobs"]
            st.session_state.progress_bar.progress(max(st.session_state.progress, 0))
            return True
    except Exception as e:
        # Create a more detailed error message
        error_obj = ErrorMessage.from_exception(e)
        error_msg = error_obj.get_truncated_message(1000) or ""
        # Add more context to the error message
        error_msg = get_more_error_details(error_msg, e)
        if processing_data:
            if "input cost $" in processing_data and processing_data["input cost $"] > 0.0:
                error_msg = f"WARNING!!! CHARGES WERE ACCRUED DURING THIS FAILED JOB:\n{processing_data = }" + error_msg
            else:
                processing_data = None    
        image_info.set_raw_llm_response(raw_llm_response=raw_response, is_associated_with_error=True)
        image_info.add_processing_data_to_image_data(processing_data)
        st.session_state.results.append({"image_info": image_info, "attempt_number": image_info.attempt_number, "status": "error", "message": f"Error processing image: {error_msg}","processing_data": processing_data})
        return False

def run_jobs():
    print(f"in run_jobs @ {get_timestamp()}")
    jobs = st.session_state.jobs_dict
    if jobs["failed"] and st.session_state.try_failed_jobs:
        jobs = load_failed_jobs(jobs) 
    while jobs["to_process"]:
        jobs["in_process"] = jobs["to_process"].pop(0)
        image_info = jobs["in_process"]
        is_successful_job = process_single_image(image_info)
        save_transcription(image_info.image_number)
        if not is_successful_job:
            move_to_failed_list(jobs, image_info)
            st.session_state.error_flag = True
            return is_successful_job
        else:    
            move_to_completed_list(jobs, image_info)    
            st.session_state.error_flag = False          
    return True

def sanitize_transcriptions(images_to_remove):
    for image_info in images_to_remove:
        image_info.delete_transcription()    

def save_cost_data():
    volume_name = st.session_state.volume_name
    model = st.session_state.io_manager.model 
    model_name = st.session_state.io_manager.model_name  
    prompt_name = st.session_state.io_manager.prompt_name
    fieldnames = st.session_state.io_manager.fieldnames
    chunk_size = st.session_state.chunk_size
    associated_transcription_filenames = st.session_state.output_files
    output_format = st.session_state.io_manager.output_format
    filename = f"{DATA_DIR}/{RUN_PREFIX}{volume_name}-data.json"
    run_numbering = st.session_state.io_manager.get_run_numbering()
    overall_costs, image_data, incomplete_jobs, completed_jobs = tally_data(run_numbering)
    input_cost_per_mil = st.session_state.io_manager.processor.input_cost_per_mil
    output_cost_per_mil = st.session_state.io_manager.processor.output_cost_per_mil
    # Create the cost data structure
    cost_data = {
        "created_by": "FieldMuseumBedrockTranscription",
        "run_id": volume_name,
        "associated_transcription_files": associated_transcription_filenames,
        "output_format": output_format,
        "chunk_size": chunk_size,
        "timestamp": datetime.datetime.now().isoformat(),
        "time_start": st.session_state.time_start,
        "model": model,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "fieldnames": fieldnames,
        "images_processed": len(completed_jobs),
        "images_failed": len(incomplete_jobs),
        "tokens": {
            "input": overall_costs["input tokens"],
            "output": overall_costs["output tokens"],
        },
        "costs": {
            "input_cost_per_mil": input_cost_per_mil,
            "output_cost_per_mil": output_cost_per_mil,
            "input": overall_costs["input cost $"],
            "output": overall_costs["output cost $"],
            "total": overall_costs["input cost $"] + overall_costs["output cost $"]
        },
        "processing_time_minutes": overall_costs["time to create/edit (mins)"]
    }
    cost_data["images"] = image_data
    cost_data["run_numbering"] = st.session_state.io_manager.get_run_numbering_as_dict()
    cost_data["completed_jobs"] = completed_jobs
    cost_data["incomplete_jobs"] = incomplete_jobs
    # Save the cost data to the JSON file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved cost data to {filename}")
    except Exception as e:
        e = ErrorMessage(e)
        print(f"Error saving cost data: {str(e)}")
        st.error(f"Error saving cost data: {str(e)}")
    return filename, cost_data

def save_transcription(image_number):
    try:
        filepath, is_saved = st.session_state.io_manager.save_transcription(image_number)
        st.session_state.save_completed = is_saved
        st.session_state.show_save_success = is_saved and not st.session_state.io_manager.error_flag
        if is_saved:
            save_cost_data()
            if filepath not in st.session_state.output_files:
                st.session_state.output_files.append(filepath)
        if st.session_state.io_manager.error_flag:
            msg = get_io_error_message()
            st.session_state.save_error_message = msg
            st.session_state.show_save_error = True        
    except Exception as e:
        e = ErrorMessage(e)
        print(f"Error in save_transcriptions_callback: {str(e)}")
        st.session_state.save_error_message = str(e)
        st.session_state.show_save_error = True

def select_and_load_run():
    saved_runs = get_saved_runs()
    print(f"{saved_runs = }")
    if not saved_runs:
        st.warning("No saved runs found.")
        return
    selected_run = st.selectbox("Select a saved run:", saved_runs)
    if st.button("Load Run"):
        missing_transcriptions = load_saved_run(selected_run)
        use_urls = "http" in missing_transcriptions[0]
        urls = missing_transcriptions if use_urls else []
        local_image_paths = missing_transcriptions if not use_urls else []
        st.session_state.process_button_clicked = True
        has_valid_input = True

def select_input_method():
    st.session_state.input_method = st.radio(
        "Choose input method:",
        ["Upload URLs File", "Select Local Images"],
        index=0,
        help="Upload a text file with URLs or select images from your local images_to_upload folder"
    )
    if st.session_state.input_method == "Upload URLs File":
        st.session_state.uploaded_file = st.file_uploader("Upload a text file with URLs (one per line)", type=["txt"])
    elif st.session_state.input_method == "Select Local Images":
        select_local_images()

def select_local_images():
    # List available images in the upload directory
    available_images = []
    for file in os.listdir(UPLOAD_IMAGES_DIR):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            available_images.append(file)
    if not available_images:
        st.warning(f"No images found in the {UPLOAD_IMAGES_DIR} folder. Please add some images and refresh.")
    else:
        if st.toggle(f"Select all images in the {UPLOAD_IMAGES_DIR} folder", value=False):
            st.session_state.selected_local_images = available_images
        else:    
            st.session_state.selected_local_images = st.multiselect(
                "Select images to process:",
                available_images,
                help=f"Select one or more images from the {UPLOAD_IMAGES_DIR} folder"
            )          
                
def select_model():
    successful_models = load_models()
    model_options = {model.get("display_name", model.get("modelId", "")): model for model in successful_models}
    st.session_state.selected_model_name = st.selectbox("Choose a model:", list(model_options.keys()))
    selected_model_obj = model_options[st.session_state.selected_model_name]
    st.session_state.selected_model = selected_model_obj.get("modelId", "")
    display_model_details(selected_model_obj)

def select_output_format():
    max_chunk_size = get_max_chunk_size(st.session_state.uploaded_file, st.session_state.selected_local_images)
    if max_chunk_size == 1:
        st.session_state.chunk_size = max_chunk_size
    else:    
        chunk_size = st.slider("Adjust Number Transcriptions per Output File", 1, max_chunk_size, max_chunk_size)
        st.session_state.chunk_size = chunk_size
        st.session_state.output_format = st.radio(
            "Choose output format:",
            ["CSV", "JSON", "TXT"],
            help="CSV: Spreadsheet format\nJSON: Structured data format\nTEXT: a single plain text file",
            index=0
        )    

def select_prompt():
    prompts = load_prompts()
    st.session_state.selected_prompt_name = st.selectbox("Choose a prompt:", list(prompts.keys()), disabled=not st.session_state.selected_model)
    st.session_state.selected_prompt_text = prompts[st.session_state.selected_prompt_name]
    display_selected_prompt_text(st.session_state.selected_prompt_text)            

def set_start_time():
    if "start_time" not in st.session_state:
        st.session_state.start_time_str = get_timestamp()

def tally_data(run_numbering):
    overall_costs = {"input tokens": 0, "output tokens": 0, "input cost $": 0.0, "output cost $": 0.0, "time to create/edit (mins)": 0.0}
    image_costs = {}
    incomplete_jobs, completed_jobs = [], []
    for image_number, image_info in run_numbering.items():
        if image_info.has_completed_transcription:
            completed_jobs.append(image_info.image_name)
        else:
            incomplete_jobs.append(image_info.image_name)
        costs = image_info.data
        if costs:
            image_costs[image_info.image_name] = costs
            for cost_name, val in costs.items():
                overall_costs[cost_name] += val
    return overall_costs, image_costs, incomplete_jobs, completed_jobs            
                    
def main():
    st.set_page_config(
    page_title="Bedrock Image Transcription App",
    page_icon="🖼️",
    layout="wide"
    )
    st.title("Bedrock Image Transcription App")
    create_directories()
    adjust_costs()
    st.session_state.show_save_success = False
    st.session_state.show_save_error = False
    set_start_time()
    with st.sidebar:
        options = get_task_options()
        st.session_state.task_option = st.radio(
            "Choose Operation", 
            options, 
            index=None, 
            key="selected_task"
        )
        if st.session_state.task_option == "Reset App":
            clear_variables()
            st.session_state.task_option = ""
            st.rerun()
        elif st.session_state.task_option == "New Run":
            st.session_state.configuration_container = st.container()
            with st.session_state.configuration_container:
                configure_inputs()
                process_button_disabled = True
                if not st.session_state.process_button_clicked and (st.session_state.uploaded_file or st.session_state.selected_local_images):
                    process_button_disabled = False
                    ###### ->                              allow for input changes if processing has not begun
                    if not st.session_state.io_manager or st.session_state.io_manager and not st.session_state.io_manager.processing_begun:
                        st.session_state.io_manager = InputOutputManager(run_name=st.session_state.volume_name, model=st.session_state.selected_model, model_name=st.session_state.model_name, prompt_name=st.session_state.selected_prompt_name, prompt_text=st.session_state.selected_prompt_text, output_format=st.session_state.output_format)
                        st.session_state.fieldnames = utils.get_fieldnames_from_prompt_text(st.session_state.selected_prompt_text)
                st.session_state.process_button_clicked = st.button("Process Images", type="primary", disabled=process_button_disabled)   
                if st.session_state.process_button_clicked:
                    st.session_state.io_manager.processing_begun = True    
        elif st.session_state.task_option == "Complete Saved Run":
            st.session_state.complete_saved_run_container = st.container()
            with st.session_state.complete_saved_run_container:
                st.header("Complete Saved Run")
                st.session_state.process_button_clicked = False
                select_and_load_run()
    ## begin processing images
    if st.session_state.process_button_clicked:
        progress = max(st.session_state.get("progress", 0), 0)
        st.session_state.progress_bar = st.progress(progress)
        status_text = st.empty()
        st.success("Processing started...")
        if st.session_state.task_option == "New Run":
            st.session_state.pre_process_inputs_container = st.container()
            with st.session_state.pre_process_inputs_container:
                pre_process_inputs()     
        # Start Setting Up Jobs
        if st.session_state.run_numbering:
            st.session_state.processing_container = st.container()
            with st.session_state.processing_container:
                begin_processing()
    if st.session_state.error_flag:
        st.session_state.error_container = st.container()
        with st.session_state.error_container:
            address_error()
    # Display Costs
    st.session_state.costs_container = st.container()
    if st.session_state.results and st.session_state.io_manager and st.session_state.io_manager.run_numbering:
        create_costs_summary()
        display_costs_summary()
    st.session_state.file_saving_success_container = st.container()
    display_file_saving_success()          
    # Display Results     
    if st.session_state.results:
        st.header("Results")
        st.session_state.success_counts_container = st.container()
        display_success_counts()
        st.session_state.results_container = st.container()  
        display_results()
        
    
if __name__ == "__main__":
    initialize_variables()
    main()