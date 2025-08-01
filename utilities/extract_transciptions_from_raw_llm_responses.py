import json
import csv
import re
import os
import sys
import utils

def extract_transcriptions(run_name, output_format):
    raw_resposes_folder = f"raw_llm_responses/{run_name}"
    output_folder = f"transcriptions/{run_name}"
    output_filepath = f"{output_folder}/{run_name}-transcriptions#1-250#{output_format}"
    usage = []
    transcriptions = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(raw_resposes_folder):
        image_name = filename.split("-raw")[0]
        file_path = os.path.join(raw_resposes_folder, filename)
        unparsed_transcription, costs = get_unparsed_transcription(file_path)
        usage.append(costs)
        transcription = clean_transcription(unparsed_transcription, image_name)
        transcriptions.append({"imageName": image_name} | transcription)
    overall_usage = sum_usage(usage)
    print(f"{overall_usage = }")
    if output_format == ".csv":
        save_to_csv(transcriptions, output_filepath)

def clean_transcription(transcription, image_name):
    #print(f"before: {image_name = }")
    #print(f"transcription = {repr(transcription)}")
    cleaned_transcription = utils.parse_innermost_dict(transcription)
    #print(f"after: {image_name}, cleaned_transcription = {repr(cleaned_transcription)}")
    return cleaned_transcription
            
def get_unparsed_transcription(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        usage = data.get("usage", {})
        transcription = data.get("output", data)
        transcription = transcription.get("message", transcription)
        transcription = transcription["content"][0] if "content" in transcription else transcription
        transcription = transcription.get("text", transcription)
        return transcription, usage

def sum_usage(usage):
    input_tokens = sum([u["inputTokens"] for u in usage])
    output_tokens = sum([u["outputTokens"] for u in usage])
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}

def save_to_csv(data, output_path):
    #print(f"{data[0] = }")
    header = data[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    print(f"Transcriptions saved to {output_path}")    
    








if __name__ == "__main__":
    run_name = "250_Segmented_Images_amazon.nova-premier-v1-2025-06-27-1432"
    output_format = ".csv"
    extract_transcriptions(run_name, output_format)


'''
sample unparsed transcription:

 {\n        \"verbatimCollectors\": \"S. Rojkowski\",\n        \"collectedBy\": \"S. Rojkowski\",\n        \"secondaryCollectors\": \"N/A\",\n        \"recordNumber\": \"394\",\n        \"verbatimEventDate\": \"8 Febr.1976\",\n        \"minimumEventDate\": \"1976-02-08\",\n        \"maximumEventDate\": \"N/A\",\n        \"verbatimIdentification\": \"Orthostichella ampuillacea (C. Müller) B. H. Allen & Magill\",\n        \"latestScientificName\": \"Orthostichella ampuillacea\",\n        \"identifiedBy\": \"R. Ochyra\",\n        \"verbatimDateIdentified\": \"N/A\",\n        \"associatedTaxa\": \"N/A\",\n        \"country\": \"Kenya\",\n        \"firstPoliticalUnit\": \"N/A\",\n        \"secondPoliticalUnit\": \"N/A\",\n        \"municipality\": \"N/A\",\n        \"verbatimLocality\": \"01 Doinyo Sapuk National Park near Thika\",\n        \"locality\": \"Doinyo Sapuk National Park near Thika\",\n        \"habitat\": \"epiphytic\",\n        \"verbatimElevation\": \"alt. 2270 m\",\n        \"verbatimCoordinates\": \"N/A\",\n        \"otherCatalogNumbers\": \"C2022783F\",\n        \"originalMethod\": \"Typed\",\n        \"typeStatus\": \"no type status\"\n    }\n}"


'''    