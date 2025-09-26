from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import openai
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv(find_dotenv())

import sys

api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
You are given a list of document titles retrieved from a knowledge base and a user query. Your task is to generate a concise title description for the chatbot. Also, generate a generic topic for the title.

CONTEXT INFORMATION:
{context}

USER QUERY:
{query}

INSTRUCTIONS:
- Limit the title description to a maximum of 5 words.

ANSWER:
title, topic
"""

def title_llm(context_text: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query="What is the best title for a chatbot based on the following documents? ")

    model = ChatOpenAI(model_name= "gpt-5", api_key=api_key)
    response_text = model.invoke(prompt)
    response = tuple(response_text.content.split(", "))
    return response[0], response[1]