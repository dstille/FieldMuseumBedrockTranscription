import json
import re

def parse_innermost_dict(data):
    if type(d) == dict and "text" in d:
        d = d["text"]
    if type(d) == dict and "transcription" in d:
        d = d["transcription"]
    return d    
    elif type(d) == str and r"{" in d:
        temp = d.split(r"{")[-1].split(r"}")[0]
        inner_dict = "{" + temp + "}" 
        try:
            return json.loads(inner_dict)
        except:
            return d 

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