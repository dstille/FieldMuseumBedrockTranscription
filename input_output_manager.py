import re
import json
import csv
import os
import time
import math
import requests
import shutil
import base64
from dotenv import load_dotenv
import bedrock_interface
from utilities.error_message import ErrorMessage
from utilities.utils import get_fieldnames_from_prompt_text

load_dotenv(override=True)

TESTING_MODE = os.getenv("TESTING_MODE", "False").lower() == "true"

class InputOutputManager:
    def __init__(self, run_name, model, model_name, prompt_name, prompt_text, output_format):
        self.processing_begun = False
        self.run_name = run_name
        self.run_numbering = {}
        self.model = model
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.prompt_text = prompt_text
        self.fieldnames = get_fieldnames_from_prompt_text(prompt_text)
        self.output_format = "." + output_format.lower()
        self.processor = self.get_image_processor()
        self.msg = {"error": []}
        self.error_flag = False
        self.run_prefix = "test-" if TESTING_MODE else ""
        self.run_folder = f"transcriptions/{self.run_prefix}{run_name}"
        self.ensure_directory_exists(self.run_folder)
        self.images_to_upload_folder = "images_to_upload"
        self.temp_images_folder = "temp_images"
        self.recovery_folder = f"recovery/{run_name}"
        self.ensure_directory_exists(self.recovery_folder)
        self.recovery_time = f"@{self.get_timestamp()}"
        print("InputOutputManager intialized")

    def copy_image(self, image_name, image_number):
        prefixed_image_name = image_name if re.match(r"\d{4}_", image_name) else f"{image_number:04d}_{image_name}"
        source_path = f"images_to_upload/{image_name}"
        dest_path = f"temp_images/{prefixed_image_name}"
        if not self.image_is_already_saved(dest_path):
            shutil.copy2(source_path, dest_path)
        return dest_path, prefixed_image_name 

    def copy_images_to_temp_folder(self, image_names):
        numbered_images = {}
        image_numbering = range(1, len(image_names)+1)
        for image_number, image_name in zip(image_numbering, image_names):
            image_path, prefixed_image_name = self.copy_image(image_name, image_number) #ImageInfo(image_number=image_number, image_name=url, local_image_name=prefixed_image_name, image_path=image_path) 
            numbered_images[image_number] = ImageInfo(image_number=image_number, image_name=image_name, local_image_name=prefixed_image_name, image_path=image_path)
        return numbered_images

    def get_image_processor(self):
        return bedrock_interface.create_image_processor(
            api_key="",  # Empty as we're using AWS credentials from environment
            prompt_name=self.prompt_name,
            prompt_text=self.prompt_text,
            model=self.model,
            modelname=self.model_name,
            output_name=self.run_name,
            testing=TESTING_MODE
        )    

    def download_image(self, url, image_number):
        image_name = url.split("/")[-1]
        prefixed_image_name = f"{image_number:04d}_{image_name}"
        image_path = os.path.join(self.temp_images_folder, prefixed_image_name)
        if self.image_is_already_saved(image_path):
            return image_path, prefixed_image_name
        response = requests.get(url)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path, prefixed_image_name
        else:
            self.error_flag = True
            self.msg["error"].append(f"Failed to download image: {url}")
            return None, None

    def download_images_to_temp_folder(self, urls):
        numbered_images = {}
        image_numbering = range(1, len(urls)+1)
        for image_number, url in zip(image_numbering, urls):
            image_path, prefixed_image_name = self.download_image(url, image_number)
            if image_path is not None:
                numbered_images[image_number] = ImageInfo(image_number=image_number, image_name=url, local_image_name=prefixed_image_name, image_path=image_path) 
        return numbered_images    

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)    

    def image_is_already_saved(self, image_path):
        return os.path.exists(image_path)

    def get_chunk(self, chunk_number):
        run_numbering = self.get_run_numbering()
        start = chunk_number * self.chunk_size + 1
        end = min(start + self.chunk_size, len(run_numbering)+1)
        return {num: image_info for num, image_info in run_numbering.items() if int(num) in range(start, end)}

    def get_gaps(self, saved_image_numbers):
        saved_image_numbers = sorted([int(num) for num in saved_image_numbers])
        latest_number_in_series = saved_image_numbers[0] - 1
        gaps = []
        for number in saved_image_numbers:
            if number != latest_number_in_series + 1:
                gaps.append(number)
                print(f"Gap found: {number = }, {latest_number_in_series = }")
            latest_number_in_series = number    
        return gaps        
                    
    def get_run_numbering(self):
        sorted_keys = sorted(self.run_numbering)
        self.run_numbering = {k: self.run_numbering[k] for k in sorted_keys}
        return self.run_numbering

    def get_run_numbering_as_dict(self):
        return {image_number: image_info.as_dict() for image_number, image_info in self.get_run_numbering().items()}     

    def get_timestamp(self):
        return time.strftime("%Y-%m-%d-%H%M")

    def load_run_numbering(self, saved_job_numbering, chunk_size):
        images_info = [val for val in saved_job_numbering.values()]
        images_to_process = [image_info["image_name"] for image_info in images_info]
        use_urls = "http" in images_to_process[0]
        run_numbering = self.set_run_numbering(images_to_process, use_urls, chunk_size, set_destination=False)
        for image_number, image_info in saved_job_numbering.items():
            run_numbering[int(image_number)].load_image_info(image_info)
        return self.get_run_numbering() 
              
    def number_run(self, set_destination):
        self.run_numbering = self.download_images_to_temp_folder(self.images_to_process) if self.use_urls else self.copy_images_to_temp_folder(self.images_to_process)
        if set_destination:
            self.set_destination_files()    
          
    def set_run_numbering(self, images_to_process, use_urls, chunk_size, set_destination=True):
        print("setting run_numbering")
        self.images_to_process = images_to_process
        self.use_urls = use_urls
        self.chunk_size = chunk_size
        self.number_run(set_destination)
        self.processing_begun = True
        return self.get_run_numbering()

    def save_transcription(self, image_number):
        chunk_number = self.run_numbering[image_number].chunk_number
        destination_file = self.run_numbering[image_number].destination_file
        filepath = f"{self.run_folder}/{destination_file}"
        recovery_file = destination_file.replace(self.output_format, f"{self.recovery_time}{self.output_format}")
        recovery_filepath = f"{self.recovery_folder}/{recovery_file}"
        images_in_chunk = self.get_chunk(chunk_number)
        saved_image_numbers = []
        if self.output_format == ".json":
            self.run_numbering[image_number].is_saved, saved_image_numbers = self.save_transcriptions_json(images_in_chunk, filepath)
            self.save_transcriptions_json(images_in_chunk, recovery_filepath)
        elif self.output_format == ".csv":
            self.run_numbering[image_number].is_saved, saved_image_numbers = self.save_transcriptions_csv(images_in_chunk, filepath)
            self.save_transcriptions_csv(images_in_chunk, recovery_filepath)
        else:
            self.run_numbering[image_number].is_saved, saved_image_numbers = self.save_transcriptions_txt(images_in_chunk, filepath)
            self.save_transcriptions_txt(images_in_chunk, recovery_filepath)   
        gaps = self.get_gaps(saved_image_numbers)
        if gaps:
            image_names = [self.run_numbering[image_number].image_name for image_number in gaps]
            self.msg["error"].append(f"Error saving transcriptions/Numbering is off: saved image numbers: {image_names}, gaps: {gaps}")
            self.error_flag = True
            print(f"Error saving transcriptions: {gaps}")    
        return filepath, self.run_numbering[image_number].is_saved
    
    def save_transcriptions_csv(self, images_in_chunk, filepath):
        transcriptions_to_save = {image_info.image_name: image_info.transcription for image_info in images_in_chunk.values() if image_info.transcription}
        image_numbers = [image_info.image_number for image_info in images_in_chunk.values() if image_info.transcription]
        saved_image_numbers = []
        try:
            fieldnames = ["imageName"]  # Start with image_name as the first field
            for image_name, data in transcriptions_to_save.items():
                if isinstance(data, dict):
                    for key in data.keys():
                        if key not in fieldnames:
                            fieldnames.append(key)
                else:
                    # If transcription is not a dict, add it as a single field
                    if "transcription" not in fieldnames:
                        fieldnames.append("transcription")
            # Write the CSV file
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for image_number, (image_name, data) in zip(image_numbers, transcriptions_to_save.items()):
                    data = {"imageName": image_name} | data if type(data) == dict else {"transcription": data}
                    for fieldname, val in data.items():
                        data[fieldname] = val.replace('\n', ' ').replace('\r', ' ')
                    data = {"imageName": image_name} | data
                    writer.writerow(data)
                    saved_image_numbers.append(image_number)
            #print(f"Successfully saved CSV transcriptions to {filepath}")
            return True, saved_image_numbers
        except Exception as e:
            e = ErrorMessage(e)
            print(f"Error saving CSV transcriptions: {str(e)}")
            self.error_flag = True
            self.msg["error"].append(f"Error saving CSV transcriptions: {str(e)}")
            return False, saved_image_numbers

    def save_transcriptions_json(self, images_in_chunk, filepath):
        transcriptions_to_save = {image_info.image_name: image_info.transcription for image_info in images_in_chunk.values() if image_info.transcription}
        image_numbers = [image_info.image_number for image_info in images_in_chunk.values() if image_info.transcription]
        saved_image_numbers = []
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(transcriptions_to_save, f, indent=2, ensure_ascii=False)
            #print(f"Successfully saved JSON transcriptions to {filepath}")
            saved_image_numbers = image_numbers
            return True, saved_image_numbers
        except Exception as e:
            e = ErrorMessage(e)
            print(f"Error saving JSON transcriptions: {str(e)}")
            self.error_flag = True
            self.msg["error"].append(f"Error saving JSON transcriptions: {str(e)}")
            return False, saved_image_numbers

    def save_transcriptions_txt(self, images_in_chunk, filepath):
        transcriptions_to_save = {image.image_name: image.transcription for image in images_in_chunk.values() if image.transcription}
        image_numbers = [image.image_number for image in images_in_chunk.values() if image.transcription]
        saved_image_numbers = []
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for i, (image_number, (image_name, data)) in enumerate(zip(image_numbers, transcriptions_to_save.items())):
                    if i > 0:
                        f.write("\n\n")
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
                    saved_image_numbers.append(image_number)
            #print(f"Successfully saved TXT transcriptions to {filepath}")
            return True, saved_image_numbers
        except Exception as e:
            e = ErrorMessage(e)
            print(f"Error saving TXT transcriptions: {str(e)}")
            self.error_flag = True
            self.msg["error"].append(f"Error saving TXT transcriptions: {str(e)}")
            return False, saved_image_numbers
        
    def set_destination_files(self):
        num_chunks = math.ceil(len(self.run_numbering) / self.chunk_size)
        chunks = [self.get_chunk(i) for i in range(num_chunks)]
        for idx, chunk in enumerate(chunks):
            beginning_number = min(chunk.keys())
            end_number = max(chunk.keys())
            for image_number in chunk:
                self.run_numbering[image_number].chunk_number = idx
                self.run_numbering[image_number].destination_file = f"{self.run_prefix}{self.run_name}#{beginning_number}-{end_number}#-transcriptions{self.output_format}"
            

###############

class ImageInfo:
    def __init__(self, image_number, image_name, local_image_name, image_path):
        self.image_number = image_number
        self.image_name = image_name
        self.local_image_name = local_image_name  # This is the name of the image in the temp folder, including the prefix and extension
        self.image_path = image_path
        self.base64_image = self.get_base64_image(image_path)
        self.attempt_number = 0
        self.has_completed_transcription = False
        self.transcription = None
        self.raw_llm_response = {}
        self.data = {}
        self.is_saved = False
        self.chunk_number = None  # This will be set by the InputOutputManager when the image is added to a chunk in set_destination_files() in input_output_manager.py
        self.destination_file = None

    def add_processing_data_to_image_data(self, processing_data):
        if not processing_data:
            return
        if self.data:
            self.data.update(processing_data)
        else:
            self.data = processing_data

    def as_dict(self):
        return self.__dict__()          

    def delete_transcription(self):
        self.transcription = None
        self.has_completed_transcription = False            

    def get_base64_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_transcription(self):
        msg = ""
        if not self.has_completed_transcription:
            if not any(v for v in transcription.values()):
                msg = "blank transcription"
            else:
                msg = "missing fieldnames"
        return self.transcription, msg            
    
    def increment_number_attempts(self):
        self.attempt_number += 1
        return self.attempt_number

    def load_image_info(self, image_info):
        self.attempt_number = image_info["attempt_number"]
        self.has_completed_transcription = image_info["has_completed_transcription"]
        self.transcription = image_info["transcription"]
        self.data = image_info["data"]
        self.is_saved = image_info["is_saved"]
        self.chunk_number = image_info["chunk_number"]
        self.destination_file = image_info["destination_file"]    

    def set_raw_llm_response(self, raw_llm_response, is_associated_with_error):
        self.raw_llm_response[self.attempt_number] = (raw_llm_response, is_associated_with_error)    

    def set_transcription(self, transcription, fieldnames):
        is_a_blank = not any(v for v in transcription.values())
        has_all_fieldnames = all(fieldname in transcription for fieldname in fieldnames)
        self.transcription = transcription
        self.has_completed_transcription = has_all_fieldnames and not is_a_blank 

    def __dict__(self):
        return {
            "image_number": self.image_number,
            "image_name": self.image_name,
            "local_image_name": self.local_image_name,
            "image_path": self.image_path,
            "attempt_number": self.attempt_number,
            "has_completed_transcription": self.has_completed_transcription,
            "transcription": self.transcription,
            "data": self.data,
            "is_saved": self.is_saved,
            "chunk_number": self.chunk_number,
            "destination_file": self.destination_file
        }
    

    def __repr__(self):
        return f"ImageInfo(image_number={self.image_number}, image_name={self.image_name}, local_image_name={self.local_image_name}, image_path={self.image_path}, number_attempts={self.attempt_number}, has_completed_transcription={self.has_completed_transcription}, transcription={self.transcription}, data={self.data}, is_saved={self.is_saved}, chunk_number={self.chunk_number}, destination_file={self.destination_file})"               

    def __str__(self):
        return f"ImageInfo(image_number={self.image_number}, image_name={self.image_name}, local_image_name={self.local_image_name}, image_path={self.image_path}, number_attempts={self.attempt_number}, has_completed_transcription={self.has_completed_transcription}, transcription={self.transcription}, data={self.data}, is_saved={self.is_saved}, chunk_number={self.chunk_number}, destination_file={self.destination_file})"        
        


            

