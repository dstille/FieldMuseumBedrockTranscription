# Field Museum Bedrock Transcription

A tool for transcribing text from images using AWS Bedrock models.

## Overview

This application allows you to transcribe text from images using various AWS Bedrock models that support image processing. The app provides a user-friendly interface for uploading images, selecting models, and obtaining transcriptions in various formats (JSON, TXT, CSV).

## Features

- **Model Selection**: Choose from models that have passed image processing tests
- **Inference Profile Support**: Automatically uses inference profiles when available
- **JSON Output**: Get transcriptions as structured JSON data
- **Error Handling**: Detailed error messages to help troubleshoot issues
- **Cost Tracking**: Monitor token usage and costs for each transcription job

## Setup

1. Clone this repository
2. Run the setup script:
   ```
   python setup.py
   ```
3. Update your AWS credentials in the `.env` file
4. Run the application:
   ```
   streamlit run app.py
   ```

## Supported Models

The application supports various models for image transcription:

### Tested and Working Models

- **Claude Models**: Claude 3.5 Sonnet, Claude 3.5 Sonnet v2, Claude 3.7 Sonnet
- **Amazon Nova Models**: Nova Pro, Nova Lite, Nova Premier
- **Meta Models**: LLaMA models (requires SageMaker endpoints)
- **Mistral Models**: Pixtral Large

## Usage

1. Select a prompt from the dropdown menu
2. Upload a text file containing image URLs (one per line)
3. Choose a model from the dropdown menu
4. Select the output format (JSON, TXT, CSV)
5. Click "Process Images" to start the transcription
6. View and download the results

## Project Structure

- `app.py`: Main Streamlit application
- `bedrock_interface.py`: Interface for AWS Bedrock models
- `llm_interface.py`: Base class for LLM interfaces
- `model_factory.py`: Factory for creating model processors
- `error_message.py`: Error handling utilities
- `base64_filter.py`: Utilities for filtering base64 content
- `vision_model_info.json`: Information about models that support image processing

## Error Handling

The application includes comprehensive error handling to help diagnose issues:

- Model compatibility checks
- Detailed error messages for API failures
- Inference profile validation
- Token usage tracking

## Data Directories

- `temp_images/`: Temporary storage for downloaded images
- `transcriptions/`: Output directory for transcriptions
- `raw_llm_responses/`: Raw responses from LLM models
- `prompts/`: Text prompts for the models
- `data/`: Cost and usage data

## Requirements

- Python 3.9+
- AWS account with Bedrock access
- Required Python packages (see requirements.txt)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Field Museum
- AWS Bedrock team
- Anthropic for Claude models
- Amazon for Nova models
- Meta for LLaMA models
- Mistral AI for Pixtral models