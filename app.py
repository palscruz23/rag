import asyncio
from utils.hybrid import hybrid_rag_bm25
import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd

# Load environment variables
load_dotenv(find_dotenv())

def main():
    st.title("Personal Chatbot")
    st.write("Enter your query below:")
    
    if "selected" not in st.session_state:
        st.session_state.selected = None
    
    query = st.text_input("Query:")
    
    options = ["RAG Vector Search", "BM25 Lexical Search", "Hybrid (RAG + BM25)"]
    selection = st.pills("Retrieval Options", options, selection_mode="single")

    if options == "RAG Vector Search":
        st.session_state.selected = ["rag"]
    elif options == "BM25 Lexical Search":
        st.session_state.selected = ["bm25"]
    elif options == "Hybrid (RAG + BM25)":
        st.session_state.selected = ["rag", "bm25"]
    else:
        st.session_state.selected = ["rag", "bm25"]

    if st.button("Submit"):
        if query:
            with st.spinner("Processing..."):
                response, sources =  hybrid_rag_bm25(query, st.session_state.selected)
                st.write("Response:")
                st.write(response)
                st.write("References:")
                st.write(sources)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()