import json

def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def fix_incomplete_jobs(data_filename):
    data = load_data(data_filename)
    run_numbering = data["run_numbering"]
    incomplete_jobs, completed_jobs = [], []
    for image_name in run_numbering:
        if not any([ext in image_name.lower() for ext in ["jpg", "jpeg", "png"]]):
            continue
        if run_numbering[image_name]["hasTranscription"]:
            completed_jobs.append(image_name)
        else:
            incomplete_jobs.append(image_name)
    data["incomplete_jobs"] = incomplete_jobs
    data["completed_jobs"] = completed_jobs
    with open(data_filename, "w") as f:
        json.dump(data, f, indent=4)    


if __name__ == "__main__":
    data_filename = "C:/Users/dancs/OneDrive/Documents/fm/June 25 Batch 1000_anthropic.claude-3-7-sonnet-20250219-v1-2025-06-25-1803-data.json"
    fix_incomplete_jobs(data_filename)
