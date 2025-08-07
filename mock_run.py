#from input_output_manager import InputOutputManager
from testing.mock_input_output_manager import MockInputOutputManager
from utilities.utils import get_fieldnames_from_prompt_text
import time
import re

def get_scenario1(run_name, model, model_name, prompt_name, prompt_text, output_format, images_to_process, use_urls, chunk_size):
    io_manager = MockInputOutputManager(run_name, model, model_name, prompt_name, prompt_text, output_format)
    fieldnames = get_fieldnames_from_prompt_text(prompt_text)
    run_numbering = io_manager.set_run_numbering(images_to_process=images_to_process, use_urls=True, chunk_size=chunk_size)
    return io_manager, fieldnames, run_numbering

def get_urls_from_file(url_filename):
    with open(url_filename, "r", encoding='utf-8') as f:
        return [url.strip() for url in f.read().splitlines()]

def get_prompt_text(prompt_filename):
    with open(f"prompts/{prompt_filename}", "r", encoding="utf-8") as f:
        return f.read()

def get_timestamp():
    return time.strftime("%Y-%m-%d-%H%M")        

def get_mock_setup():
    model = "amazon.nova-lite-v1:0"
    model_name = "amazon.nova-lite-v1"
    run_name = f"mock-{model_name}-{get_timestamp()}"
    prompt_name = "1.5Stripped.txt"
    prompt_text = get_prompt_text(prompt_name)
    output_format = "CSV"
    images_to_process = get_urls_from_file("testing/mocks/mock_runs_url_files/5-bryophytes-typed-testing-urls.txt")
    use_urls = True
    chunk_size = 3
    io_manager, fieldnames, run_numbering = get_scenario1(run_name, model, model_name, prompt_name, prompt_text, output_format, images_to_process, use_urls, chunk_size)
    return io_manager, fieldnames, run_numbering