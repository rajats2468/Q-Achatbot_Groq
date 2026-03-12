from click import prompt
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

import os

from transformers import LlamaModel
load_dotenv()

##langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = 'Q&Q Chatbot with Groq'

##Prompt tempelate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an helpful assistant.Pleas respond to the user queries"),
        ("user","Question:{Question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    llm = ChatGroq(model=llm,groq_api_key = api_key, temperature=temperature)
    chain=prompt|llm|StrOutputParser()
    answer=chain.invoke({"Question":question})
    return answer

#Title of the app
st.title("Enhanced Q&A chatbot with Groq")

##sidebar for settings
st.sidebar.title("Settings")
api_key= st.sidebar.text_input("Enter your Groq APi key:",type="password")

##Droqdown to select various groq Ai models
llm = st.sidebar.selectbox("Select a Groq Model",["qwen/qwen3-32b","llama-3.1-8b-instant","openai/gpt-oss-120b"])

##Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

##Main interface for user input
st.write("Go ahed and ask any question")
user_input=st.text_input("You:")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")

