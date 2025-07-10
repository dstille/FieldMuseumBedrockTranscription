import re
import json
import csv
import os
import time
import requests
import shutil
import base64
import bedrock_interface

TESTING_MODE = True

class InputOutputManager:
    def __init__(self, run_name, model, model_name, prompt_name, prompt_text, output_format):
        self.run_name = run_name
        self.model = model
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.prompt_text = prompt_text
        self.output_format = "." + output_format.lower()
        self.processor = self.get_image_processor()
        self.msg = {"error": []}
        self.run_folder = f"transcriptions/{run_name}"
        self.ensure_directory_exists(self.run_folder)
        self.images_to_upload_folder = "images_to_upload"
        self.temp_images_folder = "temp_images"
        self.recovery_folder = f"recovery/{run_name}"
        self.ensure_directory_exists(self.recovery_folder)
        self.recovery_time = f"@{self.get_timestamp()}"

    def copy_image(self, image_name, image_number):
        prefixed_image_name = image_name if re.match(r"\d{4}_", image_name) else f"{image_number:04d}_{image_name}"
        source_path = f"images_to_upload/{image_name}"
        dest_path = f"temp_images/{prefixed_image_name}"
        if not image_is_already_saved(dest_path):
            shutil.copy2(source_path, dest_path)
        return dest_path, prefixed_image_name 

    def copy_images_to_temp_folder(self, image_names):
        numbered_images = {}
        image_numbering = range(1, len(image_names)+1)
        for image_number, image_name in zip(image_numbering, image_names):
            image_path, prefixed_image_name = self.copy_image(image_name, image_number)
            numbered_images[image_number] = ImageInfo(image_name=image_name, local_image_name=prefixed_image_name, image_path=image_path)
        return numbered_images

    def get_image_processor(self):
        return bedrock_interface.create_image_processor(
            api_key="",  # Empty as we're using AWS credentials from environment
            prompt_name=self.prompt_name,
            prompt_text=self.prompt_text,
            model=self.model,
            modelname=self.model_name,
            output_name=self.volume_name,
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
            self.msg["error"].append(f"Failed to download image: {url}")"]
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
        start = chunk_number * self.chunk_size
        end = min(start + self.chunk_size, len(self.image_names))
        return {d for d in run_numbering if int(d.keys()[0]) in range(start, end)}

    def get_gaps(self, saved_image_numbers):
        saved_image_numbers = sorted(saved_image_numbers)
        latest_number_in_series = saved_image_numbers[0] - 1
        gaps = []
        for number in saved_image_numbers:
            if number != latest_number_in_series + 1:
                gaps.append(self.run_numbering[number])
            last_number_in_series = number    
        return gaps        
                    
    def get_run_numbering(self):
        self.run_numbering = sorted(self.run_numbering)
        return self.run_numbering 

    def get_timestamp(self):
        return time.strftime("%Y-%m-%d-%H%M") 

    def load_run_numbering(self, saved_job_numbering):
        chunk_size = saved_job_numbering["chunk_size"]
        images_to_process = [key for key in saved_job_numbering.keys() if key != "chunk_size"]
        use_urls = "http" in images_to_process[0]
        return self.set_job_numbering(images_to_process, use_urls, chunk_size) 
              
    def number_run(self):
        numbered_images = self.download_images_to_temp_folder(self.images_to_process) if use_urls else self.copy_images_to_temp_folder(self.images_to_process)
        self.run_numbering = sorted(numbered_images)
        self.set_destination_files()
          
    def set_run_numbering(self, images_to_process, use_urls, chunk_size):
        self.images_to_process = images_to_process
        self.use_urls = use_urls
        self.chunk_size = chunk_size
        self.number_run()
        return self.get_run_numbering()

    def save_transcription(self, image_number):
        chunk_number = self.run_numbering[image_number].chunk_number
        destination_file = self.run_numbering[image_number].destination_file
        filepath = f"{self.run_folder}/{destination_file}"
        recovery_file = destination_file.replace(self.output_format, f"{self.recovery_time}{self.output_format}")
        recovery_filepath = f"{self.recovery_folder}/{recovery_file}"
        images_in_chunk = self.get_chunk(chunk_number)
        if self.output_format == ".json":
            self.run_numbering[image_number].is_saved = self.save_transcriptions_json(images_in_chunk, filepath)
            self.save_transcriptions_json(images_in_chunk, recovery_filepath)
        elif self.output_format == ".csv":
            self.run_numbering[image_number].is_saved = self.save_transcriptions_csv(images_in_chunk, filepath)
            self.save_transcriptions_csv(transcriptions_to_save, recovery_filepath)
        else:
            self.run_numbering[image_number].is_saved = self.save_transcriptions_txt(images_in_chunk, filepath)
            self.save_transcriptions_txt(images_in_chunk, recovery_filepath)
        return filepath, self.run_numbering[image_number].is_saved
    
    def save_transcriptions_csv(self, images_in_chunk, filepath):
        transcriptions_to_save = {image.image_name: image.transcription for image in images_in_chunk if image.transcription}
        image_numbers = [image.image_number for image in images_in_chunk if image.transcription]
        saved_image_numbers = []
        try:
            fieldnames = ["imageName"]  # Start with image_name as the first field
            for image_number, (image_name, data) in zip(image_numbers, transcriptions_to_save.items()):
                saved_image_numbers.append(image_number)
                for fieldname, val in data.items():
                    if fieldname not in fieldnames:
                        fieldnames.append(fieldname)transcriptions.items():
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
                for image_name, data in transcriptions.items():
                    for fieldname, val in data.items():
                        data[fieldname] = val.replace('\n', ' ').replace('\r', ' ')
                    data = {"imageName": image_name} | data
                    writer.writerow(data)
                    saved_image_numbers.append(image_number)
            print(f"Successfully saved CSV transcriptions to {filepath}")
            return True, saved_image_numbers
        except Exception as e:
            print(f"Error saving CSV transcriptions: {str(e)}")
            self.msg["error"].append(f"Error saving CSV transcriptions: {str(e)}")
            return False, saved_image_numbers

    def save_transcriptions_json(self, images_in_chunk, filepath):
        transcriptions_to_save = {image.image_name: image.transcription for image in images_in_chunk if image.transcription}
        image_numbers = [image.image_number for image in images_in_chunk if image.transcription]
        saved_image_numbers = []
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(transcriptions, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved JSON transcriptions to {filepath}")
            saved_image_numbers = image_numbers
            return True, saved_image_names
        except Exception as e:
            print(f"Error saving JSON transcriptions: {str(e)}")
            self.msg["error"].append(f"Error saving JSON transcriptions: {str(e)}")
            return False, saved_image_names

    def save_transcriptions_txt(self, images_in_chunk, filepath):
        transcriptions_to_save = {image.image_name: image.transcription for image in images_in_chunk if image.transcription}
        image_numbers = [image.image_number for image in images_in_chunk if image.transcription]
        saved_image_numbers = []
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for image_number, (i, (image_name, data)) in zip(image_numbersenumerate(transcriptions.items())):
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
                    saved_image_numbers.append(image_number)
            print(f"Successfully saved TXT transcriptions to {filepath}")
            return True, saved_image_numbers
        except Exception as e:
            print(f"Error saving TXT transcriptions: {str(e)}")
            self.msg["error"].append(f"Error saving TXT transcriptions: {str(e)}")
            return False, saved_image_numbers
        
    def set_destination_files(self):
        chunks = [self.get_chunk(i) for i in range((len(self.run_numbering) - 1) // self.chunk_size + 1)]
        for idx, chunk in enumerate(chunks):
            beginning_number = min(chunk.keys())
            end_number = max(chunk.keys())
            for image_number in chunk:
                self.run_numbering[image_number].chunk_number = idx
                self.run_numbering[image_name].destination_file = f"{self.run_name}#{beginning_number}-{end_number}#-transcriptions{self.output_format}"

###############

class ImageInfo:
    def __init__(self, image_number, image_name, local_image_name, image_path):
        self.image_number = image_number
        self.image_name = image_name
        self.local_image_name = local_image_name  # This is the name of the image in the temp folder, including the prefix and extension
        self.image_path = image_path
        self.base64_image = get_base64_image(image_path)
        self.number_attempts = 0
        self.has_completed_transcription = False
        self.transcription = None
        self.image_data = {}
        self.is_saved = False
        self.chunk_number = None  # This will be set by the InputOutputManager when the image is added to a chunk in set_destination_files() in input_output_manager.py
        self.destination_file = None

    def add_processing_data_to_image_data(self, processing_data):
        if not processing_data:
            return
        if self.image_data:
            self.image_data.update(processing_data)
        else:
            self.image_data = processing_data 

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
        self.number_attempts += 1
        return self.number_attempts

    def set_transcription(self, transcription, fieldnames):
        is_a_blank = not any(v for v in transcription.values())
        has_all_fieldnames = all(fieldname in transcription for fieldname in fieldnames)
        self.transcription = transcription
        self.has_completed_transcription = has_all_fieldnames and not is_a_blank   

    def __str__(self):
        return f"ImageInfo(image_number={self.image_number}, image_name={self.image_name}, local_image_name={self.local_image_name}, image_path={self.image_path}, number_attempts={self.number_attempts}, has_completed_transcription={self.has_completed_transcription}, transcription={self.transcription}, image_data={self.image_data}, is_saved={self.is_saved}, chunk_number={self.chunk_number}, destination_file={self.destination_file})"        
        


            

