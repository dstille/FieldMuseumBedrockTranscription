import streamlit as st
import base64
import boto3
import json
import os
from PIL import Image

def analyze_image(image_file, prompt, format, model_id):
    # Create a Bedrock Runtime client
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
    )
    
    # Convert the uploaded file to base64
    binary_data = image_file.getvalue()
    base_64_encoded_data = base64.b64encode(binary_data)
    base64_string = base_64_encoded_data.decode("utf-8")

    # Convert format to lowercase for API compatibility
    format = format.lower()

    # Configure request based on model type
    if "claude" in model_id.lower():
        # Claude-specific request format
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{format}",
                                "data": base64_string
                            }
                        },
                        {
                            "type": "text",
                            "text": f"{prompt}\nProvide art titles for this image."
                        }
                    ]
                }
            ]
        }
    else:
        # Nova/default model format
        # Define system prompt(s)
        system_list = [{
            "text": prompt
        }]

        # Define message list
        message_list = [{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": format,
                        "source": {"bytes": base64_string},
                    }
                },
                {
                    "text": "Provide art titles for this image."
                }
            ],
        }]

        # Configure inference parameters
        inf_params = {"max_new_tokens": 300, "top_p": 0.1, "top_k": 20, "temperature": 0.3}

        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inf_params,
        }

    # Invoke the model
    response = client.invoke_model(modelId=model_id, body=json.dumps(request_body))
    model_response = json.loads(response["body"].read())
    
    return model_response

def main():
    st.title("Image Analysis with Amazon Bedrock")
    st.write("Upload an image and get AI-generated art titles!")

    # Model selector
    model_options = {
        "Nova Lite": "us.amazon.nova-lite-v1:0",
        "Nova": "us.amazon.nova-v1:0",
        "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
    
    selected_model_name = st.selectbox(
        'Select AI model',
        list(model_options.keys())
    )
    
    # Get the model ID from the selected name
    model_id = model_options[selected_model_name]

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    # Text input for prompt
    prompt = st.text_input("Enter your prompt", "You are an art curator. Analyze this image.")
    
    # Format selector (keeping display capitalized for UI but will convert to lowercase in function)
    format_option = st.selectbox(
        'Select image format',
        ('PNG', 'JPEG')
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Analyze button
        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Get results
                    results = analyze_image(uploaded_file, prompt, format_option, model_id)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Handle different response formats based on model
                    if "claude" in model_id.lower():
                        content_text = results["content"][0]["text"]
                    else:
                        content_text = results["output"]["message"]["content"][0]["text"]
                    
                    st.write(content_text)
                    
                    # Optional: Display full response in expander
                    with st.expander("See full response"):
                        st.json(results)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()