#!/usr/bin/env python3
"""
Utility script to save JSON responses from Bedrock transcriptions.
This script can convert existing text transcriptions to JSON format.
"""

import os
import json
import argparse
import re
from pathlib import Path

def ensure_directory_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_json_from_text(text):
    """
    Extract JSON from text response.
    Handles cases where the model might include text before or after the JSON.
    """
    try:
        # First try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON using regex
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return the original text as a JSON object
        return {"raw_text": text, "error": "Could not parse as JSON"}

def convert_transcription_to_json(input_file, output_file=None):
    """
    Convert a text transcription to JSON format.
    
    Args:
        input_file: Path to the text transcription file
        output_file: Path to save the JSON file (default: same name with .json extension)
    
    Returns:
        Path to the saved JSON file
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract JSON from text
    json_data = extract_json_from_text(text)
    
    # Determine output file path
    if not output_file:
        output_file = str(Path(input_file).with_suffix('.json'))
    
    # Save JSON to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return output_file

def batch_convert_transcriptions(input_dir, output_dir=None):
    """
    Convert all text transcriptions in a directory to JSON format.
    
    Args:
        input_dir: Directory containing text transcription files
        output_dir: Directory to save JSON files (default: same as input_dir)
    
    Returns:
        Number of files converted
    """
    # Determine output directory
    if not output_dir:
        output_dir = input_dir
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Get all text files in the input directory
    input_path = Path(input_dir)
    count = 0
    
    for file_path in input_path.glob('*.txt'):
        output_file = Path(output_dir) / f"{file_path.stem}.json"
        try:
            convert_transcription_to_json(str(file_path), str(output_file))
            count += 1
            print(f"Converted {file_path} to {output_file}")
        except Exception as e:
            print(f"Error converting {file_path}: {str(e)}")
    
    return count

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert text transcriptions to JSON format"
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input file or directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to output file or directory (default: auto-generated)"
    )
    
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Process all files in the input directory"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Check if input path exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input path does not exist: {args.input_path}")
        return 1
    
    # Process batch mode
    if args.batch:
        if not os.path.isdir(args.input_path):
            print(f"Error: Input path must be a directory in batch mode: {args.input_path}")
            return 1
        
        count = batch_convert_transcriptions(args.input_path, args.output)
        print(f"Converted {count} files")
        
    else:  # Single file mode
        if not os.path.isfile(args.input_path):
            print(f"Error: Input path must be a file in single mode: {args.input_path}")
            return 1
        
        try:
            output_file = convert_transcription_to_json(args.input_path, args.output)
            print(f"Converted {args.input_path} to {output_file}")
        except Exception as e:
            print(f"Error converting file: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())