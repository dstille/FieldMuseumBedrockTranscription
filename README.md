# Field Museum Bedrock Transcription

A tool for transcribing herbarium specimen labels using AWS Bedrock models.

## Setup

1. Run the setup script to create necessary directories and install requirements:
   ```
   python setup.py
   ```

2. Configure your AWS credentials in the `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_access_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_key_here
   AWS_REGION=us-east-1
   ```

3. Place your prompt files in the `prompts` directory.

## Running the App

Start the Streamlit app:
```
streamlit run app.py
```

## Testing Models

To test all image-capable models with a single image:

1. Place a test image in the `test_images` directory
2. Update the `IMAGE_PATH` and `PROMPT_FILE` variables in `models_image_test.py` if needed
3. Run the test script:
   ```
   python models_image_test.py
   ```

Test results will be saved in the `test_results` directory.

## Directory Structure

- `models/`: Model-specific processor classes
- `prompts/`: Prompt templates
- `transcriptions/`: Saved transcription outputs
- `data/`: Cost and usage data
- `test_images/`: Images for model testing
- `test_results/`: Model comparison results
- `temp_images/`: Temporary storage for downloaded images
- `raw_llm_responses/`: Raw responses from LLM models

## Output Formats

Transcriptions can be saved in three formats:
- JSON: Structured data format
- TXT: Plain text format
- CSV: Spreadsheet format

## Cost Tracking

Cost data for each run is saved in the `data` directory as `{volume_name}-data.json`.