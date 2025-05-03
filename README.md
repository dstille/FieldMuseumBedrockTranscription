# Bedrock Image Transcription App

A Streamlit application that processes images from URLs using AWS Bedrock models.

## Setup

1. Install required dependencies:
   ```
   pip install streamlit boto3 requests
   ```

2. Configure AWS credentials:
   - Set environment variables:
     ```
     AWS_ACCESS_KEY_ID=your_access_key
     AWS_SECRET_ACCESS_KEY=your_secret_key
     AWS_REGION=us-east-1  # or your preferred region
     ```
   - Or configure using AWS CLI:
     ```
     aws configure
     ```

3. Update available models:
   ```
   python available_models.py
   ```
   This will update the `selected_models.json` file with models available in your AWS account.

## Running the App

```
streamlit run app.py
```

## Usage

1. Select a prompt from the prompts folder
2. Upload a text file containing image URLs (one per line)
3. Select a Bedrock model
4. Click "Process Images"

## Troubleshooting

### Model Invocation Errors

If you encounter errors like:
```
ValidationException: Invocation of model ID ... with on-demand throughput isn't supported
```

This means the selected model doesn't support on-demand throughput. Try:

1. Running `python available_models.py` to update the model list
2. Selecting a different model
3. Creating an inference profile for the model in the AWS Bedrock console

### Authentication Errors

If you see authentication errors, check:
1. Your AWS credentials are correctly set
2. Your AWS account has access to AWS Bedrock
3. You have enabled the models you want to use in the AWS Bedrock console

## Folders

- `prompts/`: Contains text prompts for image processing
- `temp_images/`: Temporary storage for downloaded images
- `transcriptions/`: Output folder for transcription results
- `raw_llm_responses/`: Raw JSON responses from the LLM models