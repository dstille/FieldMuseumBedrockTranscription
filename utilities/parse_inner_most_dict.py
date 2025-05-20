import json
import re

def parse_innermost_dict(data):
    if type(d) == dict:
        d = d["transcription"] if "transcription" in d else d["text"] if "text" in d else d
        return "\n".join(f"{k}: {v}" for k, v in d.items())
    elif type(d) == str and r"{" in d:
        inner_dict = d.split(r"{")[-1].split(r"}")[0]
        temp = re.sub(r"[\n\'\"]", "", inner_dict)
        lines = [line.strip() for line in temp.split(",")]
        return "\n".join(lines)
    else:
        return json.dumps(d)  

def parse_from_string(data):
    pattern = r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)'
    matches = re.findall(pattern, data)
    print(f"{matches = }")
    if matches:
        return matches
    else:
        return data    


def parse(data):
    if type(data) == dict:
        innermost = parse_from_dict(data)
    elif type(data) == str:
        innermost = parse_from_string(data) 
    else:
        innermost = data
    print(f"{innermost = }")           

if __name__ == '__main__':
    with open('transcriptions/nova-2025-05-13-1825.json', 'r', encoding='utf-8') as f:
        data = json.load(f)