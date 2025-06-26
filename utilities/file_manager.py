import re
import json
import csv
import os
import time

class FileManager:
    def __init__(self, run_name, output_format):
        self.run_name = run_name
        self.output_format = "." + output_format.lower()
        self.run_folder = f"transcriptions/{run_name}"
        self.ensure_directory_exists(self.run_folder)
        self.recovery_folder = f"recovery/{run_name}"
        self.ensure_directory_exists(self.recovery_folder)
        self.recovery_time = f"@{self.get_timestamp()}"

    def set_run_numbering(self, image_names, chunk_size):
        self.image_names = image_names
        self.chunk_size = chunk_size
        self.number_images()
        return self.run_numbering

    def load_run_numbering(self, run_numbering):
        self.chunk_size = run_numbering["chunk_size"]
        self.image_names = [key for key in run_numbering.keys() if key != "chunk_size"]
        self.run_numbering = run_numbering
        #self.set_destination_files()
        return self.run_numbering
    
    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_chunk(self, chunk_number):
        start = chunk_number * self.chunk_size
        end = min(start + self.chunk_size, len(self.image_names))
        return self.image_names[start:end]

    def get_timestamp(self):
        return time.strftime("%Y-%m-%d-%H%M")
    
    def number_images(self):
        self.run_numbering = {image_name: {"imageNumber": idx + 1, "numberAttempts": 0, "hasTranscription": False, "destination_file": None, "is_saved": False} for idx, image_name in enumerate(self.image_names)}
        self.set_destination_files()

    def save_transcription(self, image_name, transcriptions):
        chunk_number = self.run_numbering[image_name]["chunk_number"]
        destination_file = self.run_numbering[image_name]["destination_file"]
        filepath = f"{self.run_folder}/{destination_file}"
        recovery_file = destination_file.replace(self.output_format, f"{self.recovery_time}{self.output_format}")
        recovery_filepath = f"{self.recovery_folder}/{recovery_file}"
        images_in_chunk = self.get_chunk(chunk_number)
        sorted_image_names = sorted(images_in_chunk, key=lambda x: self.run_numbering[x]["imageNumber"])
        transcriptions_to_save = {image_name: transcriptions[image_name] for image_name in sorted_image_names if image_name in transcriptions}
        if self.output_format == ".json":
            self.run_numbering[image_name]["is_saved"] = self.save_transcriptions_json(transcriptions_to_save, filepath)
            self.save_transcriptions_json(transcriptions_to_save, recovery_filepath)
        elif self.output_format == ".csv":
            self.run_numbering[image_name]["is_saved"] = self.save_transcriptions_csv(transcriptions_to_save, filepath)
            self.save_transcriptions_csv(transcriptions_to_save, recovery_filepath)
        else:
            self.run_numbering[image_name]["is_saved"] = self.save_transcriptions_txt(transcriptions_to_save, filepath)
            self.save_transcriptions_txt(transcriptions_to_save, recovery_filepath)
        return filepath, self.run_numbering[image_name]["is_saved"]
    
    def save_transcriptions_csv(self, transcriptions, filepath):
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
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for image_name, data in transcriptions.items():
                    for fieldname, val in data.items():
                        data[fieldname] = val.replace('\n', ' ').replace('\r', ' ')
                    data = {"imageName": image_name} | data
                    writer.writerow(data)
            print(f"Successfully saved CSV transcriptions to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving CSV transcriptions: {str(e)}")
            return False

    def save_transcriptions_json(self, transcriptions, filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(transcriptions, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved JSON transcriptions to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving JSON transcriptions: {str(e)}")
            return False

    def save_transcriptions_txt(self, transcriptions, filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
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
            print(f"Successfully saved TXT transcriptions to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving TXT transcriptions: {str(e)}")
            return False
        
    def set_destination_files(self):
        self.run_numbering["chunk_size"] = self.chunk_size
        chunks = [self.get_chunk(i) for i in range((len(self.image_names) - 1) // self.chunk_size + 1)]
        for idx, chunk in enumerate(chunks):
            beginning_number = self.run_numbering[chunk[0]]["imageNumber"]
            end_number = self.run_numbering[chunk[-1]]["imageNumber"]
            for image_name in chunk:
                self.run_numbering[image_name]["chunk_number"] = idx
                self.run_numbering[image_name]["destination_file"] = f"{self.run_name}-transcriptions#{beginning_number}-{end_number}#{self.output_format}"
###############

            

