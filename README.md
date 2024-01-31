# Shunya.ek Chatbot (pari) v1.0

## Overview
This is a Streamlit-based chatbot designed to assist users in understanding and navigating the 'pari' product documentation. The chatbot utilizes contextual embeddings and the GPT-3.5-turbo model from OpenAI to provide informative and relevant responses.

## Dependencies
- streamlit
- pandas
- scipy
- openai
- tiktoken
- ast
- numpy

## Installation
To run the chatbot, make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install streamlit pandas scipy openai tiktoken ast numpy
```

## Usage

1. Replace the placeholder data file ('pariembeded.csv') with your own data file.
2. Set your OpenAI API key in the OpenAI initialization section.
3. Run the script.

## Functions

### strings_ranked_by_relatedness
Returns a list of strings and relatedness scores, sorted from most related to least.

### num_tokens
Returns the number of tokens in a given string using the specified model.

### query_message
Prepares a message for GPT by selecting relevant source texts from the dataframe based on relatedness to the user's query.

### ask
Utilizes GPT to provide a response to the user's query using the specified dataframe and model.

## Streamlit App

The app provides a simple interface to interact with the chatbot.
- Users can input questions and receive responses based on the 'pari' product documentation.
- Conversation history is displayed below the input, showing both user queries and bot responses.

## Note

- Ensure that you comply with OpenAI's usage policies and guidelines when using the GPT-3.5-turbo model.
- Customize the conversation instructions and context based on your specific use case and requirements.
