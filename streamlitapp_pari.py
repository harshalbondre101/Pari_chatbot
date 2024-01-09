import streamlit as st
import pandas as pd
from scipy import spatial
from openai import OpenAI
import tiktoken
import ast
import numpy as np
# Load data
df = pd.read_csv('pariembeded.csv')  # Replace with your data file
df['embedding'] = df['Embedding'].apply(ast.literal_eval)

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# OpenAI initialization
openai = OpenAI(api_key="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")

# Initialize session state for storing conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to rank strings by relatedness to a query

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = np.array(query_embedding_response.data[0].embedding)

    def normalize_embedding(embedding):
        return np.array(embedding).flatten()

    strings_and_relatednesses = [
        (
            row["one"],
            relatedness_fn(
                normalize_embedding(query_embedding),
                normalize_embedding(row["embedding"])
            )
        )
        for _, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


# Function to count tokens in a string
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to prepare a message for GPT with relevant source texts
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = """As a conversational AI, your primary goal is to assist users in understanding their queries using contextual embeddings. Act naturally and engage users by addressing their questions, while disregarding any irrelevant information within the context. Your role is to seamlessly navigate through the given data, focusing solely on the user's inquiries. Start by answering the questions and avoid referencing or acknowledging irrelevant details within the context."""
    message = introduction
    
    question = f"\n\nQuestion: {query}"
    for string in strings:
        next_article = f'\n\n Context data:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    
    return message + question

# Function to interact with GPT and provide a response
def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
    conversation_history: list = []
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "Your role is that of an AI chatbot utilizing exclusively the 'pari' product documentation to respond to queries regarding 'shunya.ek'. Your task involves addressing user questions in a friendly, factual, and conversational manner, adhering strictly to the information contained within the 'pari' documentation. Offer step-by-step guidance, explanations, and assistance as needed to cater to user needs. Emphasize accuracy and refrain from generating information beyond the scope of the provided context. Your responses must align with the 'shunya.ek' subject matter, utilizing the content available within the 'pari' product documentation to create outputs that are informative and pertinent to user inquiries."},
        {"role": "user", "content": message},
    ] + conversation_history

    if print_message:
        print(message)

    # Get GPT response
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

# Streamlit App for conversation history
st.title("Shunya.ek Chatbot version 1.0 (pari)")
st.text("You can ask any question about pari and get answers based on the documentation. ")

conversation_history = []

# Text input for user queries
query = st.text_input("", placeholder="Enter your question:")

# Ask button triggers query processing
if st.button("Ask"):
    response = ask(query, conversation_history=conversation_history)
    # Append user query and bot response to conversation history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "bot", "content": response})
    query = ""  # Clear the text input after clicking the "Ask" button

# Display conversation history
conversation_output = ""
for message in st.session_state.messages:
    if message["role"] == "user":
        conversation_output += f"**You**: {message['content']}  \n"
    else:
        conversation_output += f"Bot: {message['content']}  \n \n \n"

st.markdown(conversation_output)

