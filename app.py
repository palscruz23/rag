import asyncio
from utils.hybrid import hybrid_rag_bm25
import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import utils.populate_database
import sys
import shutil
# Load environment variables
load_dotenv(find_dotenv())


def main():
    if "selected" not in st.session_state:
        st.session_state.selected = None

    st.title("Personal Chatbot")

    with st.sidebar:

        if "uploaded_file" not in st.session_state:
            st.session_state.uploaded_file = None

        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file is not None:
            upload = st.button("Upload pdf")
            if upload: 
                # Save the file locally
                with open(f"./knowledge base/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved file: {uploaded_file.name}")

                utils.populate_database.main()
                st.success("Database updated!")
                
                # Clear uploader by resetting session state and rerunning
                st.session_state.uploaded_file = None
                st.rerun()

        # Folder to list files from
        folder_path = "knowledge base/"
        os.makedirs(folder_path, exist_ok=True)  # create folder if it doesn't exist

        # Get list of files
        files = os.listdir(folder_path)

        # Show in sidebar
        st.sidebar.header("Files in Knowledge Base")
        if files:
            markdown_list = "\n".join([f"- {file}" for file in files])
            st.sidebar.markdown(markdown_list)
        else:
            st.sidebar.write("No files found")

            
        # Initialize session state flags
        if "db_reset" not in st.session_state:
            st.session_state.db_reset = False

        if "reset" not in st.session_state:
            st.session_state.reset = False

        # Reset button
        if st.button("Reset Database"):
            st.session_state.reset = True
            st.session_state.db_reset = False

        # Only run reset logic **once**
        if st.session_state.reset and not st.session_state.db_reset:
            # Your database cleanup logic here
            
            
            # Example: delete all files under folder
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)

            # Example: delete all vectors from Chroma
            from langchain_chroma import Chroma
            from utils.get_embeddings import get_embedding_function
            embedding_function = get_embedding_function()
            
            CHROMA_PATH =  "./utils/chroma"
            client = Chroma(
                                persist_directory=CHROMA_PATH,
                                embedding_function=get_embedding_function()
                            )
            
            # collections = client.list_collections()
            existing_items = client.get(include=[])

            existing_ids = set(existing_items["ids"])

            # Delete all existing vectors
            if existing_ids:
                client.delete(ids=list(existing_ids))
                st.write(f"Number of existing documents in DB: {len(existing_ids)}")

            # Mark as done
            st.session_state.db_reset = True
            st.session_state.reset = False
            st.success("Database reset completed!")
            st.rerun()

            
    if files:
        st.write("Enter your query below:")
        query = st.text_input("Query:")
        options = ["RAG Vector Search", "BM25 Lexical Search", "Hybrid (RAG + BM25)"]
        selection = st.pills("Retrieval Options", options, selection_mode="single")
        if selection == "RAG Vector Search":
            st.session_state.selected = ["rag"]
        elif selection == "BM25 Lexical Search":
            st.session_state.selected = ["bm25"]
        elif selection == "Hybrid (RAG + BM25)":
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
    else:
        st.info("No documents in the knowledge base. Please upload a PDF in the sidebar.")

if __name__ == "__main__":
    # if sys.argv[-1] == "reset":
    #     utils.populate_database.clear_database()
    #     folder_path = "./knowledge base/"
    #     for filename in os.listdir(folder_path):
    #         file_path = os.path.join(folder_path, filename)
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    main()