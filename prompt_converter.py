#!/usr/bin/env python3
"""
Prompt Converter Tool for Field Museum Bedrock Transcription application.
This script converts text prompts to JSON format and vice versa.
"""

import os
import sys
import argparse
from pathlib import Path
from utils import (
    prompt_to_json, 
    json_to_prompt, 
    read_text_file, 
    save_json, 
    load_json, 
    write_text_file,
    ensure_directory_exists,
    batch_convert_prompts
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert between text prompts and JSON format"
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
        "-d", "--direction",
        choices=["to_json", "to_text"],
        default="to_json",
        help="Conversion direction (default: to_json)"
    )
    
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Process all files in the input directory"
    )
    
    return parser.parse_args()

def convert_single_file(input_path, output_path, direction):
    """Convert a single file between text and JSON formats"""
    try:
        if direction == "to_json":
            # Convert text prompt to JSON
            prompt_text = read_text_file(input_path)
            prompt_json = prompt_to_json(prompt_text)
            
            # Generate output path if not provided
            if not output_path:
                output_path = str(Path(input_path).with_suffix(".json"))
            
            # Save JSON
            save_json(prompt_json, output_path)
            print(f"Converted text prompt to JSON: {output_path}")
            
        else:  # to_text
            # Convert JSON to text prompt
            prompt_json = load_json(input_path)
            prompt_text = json_to_prompt(prompt_json)
            
            # Generate output path if not provided
            if not output_path:
                output_path = str(Path(input_path).with_suffix(".txt"))
            
            # Save text
            write_text_file(prompt_text, output_path)
            print(f"Converted JSON to text prompt: {output_path}")
            
    except Exception as e:
        print(f"Error converting file {input_path}: {str(e)}")
        return False
    
    return True

def main():
    """Main function"""
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
        
        # Set default output directory if not provided
        output_dir = args.output if args.output else f"{args.input_path}_converted"
        ensure_directory_exists(output_dir)
        
        # Convert all files in the directory
        batch_convert_prompts(
            args.input_path, 
            output_dir, 
            to_json=(args.direction == "to_json")
        )
        
    else:  # Single file mode
        if not os.path.isfile(args.input_path):
            print(f"Error: Input path must be a file in single mode: {args.input_path}")
            return 1
        
        # Convert the file
        success = convert_single_file(args.input_path, args.output, args.direction)
        if not success:
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())