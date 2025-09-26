from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import find_dotenv, load_dotenv
import os
import streamlit as st
# Load environment variables
load_dotenv(find_dotenv())
# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def get_embedding_function():
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY']
)
    return embeddings