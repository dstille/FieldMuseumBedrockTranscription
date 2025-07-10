import csv
import json
import re


def extract_index_from_image_name(image_name):
    match = re.match(r'(\d+)_', image_name)
    return int(match.group(1)) if match else float('inf')

def sort_dicts_by_image_name(dicts):
    return sorted(dicts, key=lambda x: extract_index_from_image_name(x['imageName']))   

def load_data(filepath):
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def save_data(data, filepath):
    with open(filepath, "w", newline= "", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)        

def reorder_data(filepaths):
    for filepath in filepaths:
        data = load_data(filepath)
        sorted_data = sort_dicts_by_image_name(data)
        print(list(sorted_data[0].values()))
        save_data(sorted_data, filepath)      

if __name__ == "__main__":
    filepaths = ["C:/Users/dancs/OneDrive/Documents/GitHub/FieldMuseumTranscription/DataAnalysis/RileyImageManipulation/Transcriptions/250_Segmented_Images_amazon.nova-premier-v1-2025-06-27-1432-transcriptions.csv", "C:/Users/dancs/OneDrive/Documents/GitHub/FieldMuseumTranscription/DataAnalysis/RileyImageManipulation/Transcriptions/250_Segmented_Images_anthropic.claude-3-7-sonnet-20250219-v1-2025-06-27-0913-transcriptions.csv", "C:/Users/dancs/OneDrive/Documents/GitHub/FieldMuseumTranscription/DataAnalysis/RileyImageManipulation/Transcriptions/amazon.nova-pro-v1-2025-06-27-1344-transcriptions.csv", "C:/Users/dancs/OneDrive/Documents/GitHub/FieldMuseumTranscription/DataAnalysis/RileyImageManipulation/Transcriptions/June 27 batch 250 non-segmented_anthropic.claude-3-7-sonnet-20250219-v1-2025-06-27-0948-transcriptions.csv"]
    reorder_data(filepaths)
        
            
    