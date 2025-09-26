import asyncio
from utils.hybrid import hybrid_rag_bm25
from utils.title_generator import title_llm
import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import utils.populate_database
import sys
import shutil
# Load environment variables
load_dotenv(find_dotenv())

def init_chatbot():
    
    st.sidebar.header("Conversation Options")
    st.sidebar.button("Clear conversation", key="clear_conversation")
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)

    # if "topic" not in st.session_state:
    #     st.session_state.topic = "anything in the PDF files"

    if "title" not in st.session_state:
        folder_path = "knowledge base/"
        os.makedirs(folder_path, exist_ok=True)  # create folder if it doesn't exist
        files = os.listdir(folder_path)
        if files != []:
            title, topic = title_llm(str(files))
            st.session_state.title = title
            st.session_state.topic = topic
        else:
            st.session_state.title = "Personal Chatbot"
            st.session_state.topic = "anything in the PDF files"
    
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []


## Get the chat history
def get_chat_history():
    start_index = max(
        0, len(st.session_state.messages) - st.session_state.num_chat_messages
    )
    return st.session_state.messages[start_index : len(st.session_state.messages) - 1]

def make_chat_history_summary(chat_history, question):
    
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extend the question with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """
    return prompt

def main():
    init_chatbot()

    st.title(st.session_state.title)
    icons = {"assistant": "❄️", "user": "👤"}

    if "selected" not in st.session_state:
        st.session_state.selected = None

    with st.sidebar:

        if "uploaded_file" not in st.session_state:
            st.session_state.uploaded_file = None
        st.sidebar.header("Upload PDF")

        uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        pdf_files = []

        if uploaded_files:
            upload = st.button("Upload pdf")
            if upload: 
                for uploaded_file in uploaded_files:
                    # Save the file locally
                    pdf_files.append(uploaded_file.name)
                    with open(f"./knowledge base/{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                utils.populate_database.main()
                st.success("Database updated!")
                
                # Clear uploader by resetting session state and rerunning
                st.session_state.uploaded_file = True
                folder_path = "knowledge base/"
                files = os.listdir(folder_path)
                if files != []:
                    title, topic = title_llm(str(files))
                    st.session_state.title = title
                    st.session_state.topic = topic
                else:
                    st.session_state.title = "Personal Chatbot"
                    st.session_state.topic = "anything in the PDF files"
                st.rerun()

        # Folder to list files from
        folder_path = "knowledge base/"
        os.makedirs(folder_path, exist_ok=True)  # create folder if it doesn't exist

        # Get list of files
        files = os.listdir(folder_path)

        # State to control display mode
        if "show_radio" not in st.session_state:
            st.session_state.show_radio = False

        # Initialize session state flags
        if "reset" not in st.session_state:
            st.session_state.reset = False

        # Reset button
        if st.button("Remove files"):
            st.session_state.show_radio = not st.session_state.show_radio
            st.session_state.reset = True

        # Show in sidebar
        st.sidebar.header("Files in Knowledge Base")
        if files:
            if st.session_state.show_radio:
                # Show radio buttons
                selected_file = st.sidebar.radio("Select a file:", files + ["All files"])
                st.sidebar.write(f"Selected: {selected_file}")
            else:
                # Show static markdown list
                markdown_list = "\n".join([f"- {file}" for file in files])
                st.sidebar.markdown(markdown_list)
        else:
            st.sidebar.write("No files found")

        # Only run reset logic 
        if st.session_state.reset:
            if st.button("Select file to delete"):
                st.session_state.show_radio = not st.session_state.show_radio
                # Your database cleanup logic here
                if selected_file != "All files":
                    file_path = folder_path + selected_file
                    files = [selected_file]
                    if os.path.exists(file_path):
                        os.remove(file_path)   # deletes the file
                        print(f"Deleted: {file_path}")
                    else:
                        print("File does not exist")
                else:
                    # Delete all files under folder
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                    os.makedirs(folder_path, exist_ok=True)
                    files = os.listdir(folder_path)

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
                # (Optional) verify
                # print("Remaining IDs:", existing_items["ids"])
                # st.write("Remaining IDs:", existing_items["ids"])
                for file in files:
                    file_path_id = "knowledge base\\" + file
                    ids_to_delete = [
                        doc_id
                        for doc_id in zip(existing_items["ids"])
                        if file_path_id in doc_id]
                    # Delete all existing vectors
                    if ids_to_delete:
                        client.delete(ids=list(ids_to_delete))

                # Mark as done
                st.session_state.db_reset = True
                st.session_state.reset = False
                st.session_state.title = "Personal Chatbot"
                st.success("Database reset completed!")
                st.rerun()
        
        st.sidebar.header("Retrieval Method")
        options = ["RAG Vector Search", "BM25 Lexical Search", "Hybrid (RAG + BM25)"]
        selection = st.radio("Retrieval Options", options, label_visibility="collapsed")
        if selection == "RAG Vector Search":
            st.session_state.selected = ["rag"]
        elif selection == "BM25 Lexical Search":
            st.session_state.selected = ["bm25"]
        elif selection == "Hybrid (RAG + BM25)":
            st.session_state.selected = ["rag", "bm25"]
        else:
            st.session_state.selected = ["rag", "bm25"]

            
    if files:

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=icons[message["role"]]):
                st.markdown(message["content"])

        if question := st.chat_input("Any talks about " + str(st.session_state.topic)):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            # Display user message in chat message container
            with st.chat_message("user", avatar=icons["user"]):
                st.markdown(question.replace("$", r"\$"))

            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar=icons["assistant"]):
                message_placeholder = st.empty()
                question = question.replace("'", "")
                with st.spinner("Thinking..."):
                    if st.session_state.use_chat_history:
                        chat_history = get_chat_history()
                        if chat_history == []:
                            hist_prompt = ""
                        else:
                            hist_prompt = make_chat_history_summary(chat_history, question)
                    else:
                        hist_prompt = ""
                    response, sources =  hybrid_rag_bm25(question, st.session_state.selected, hist_prompt)
                    message_placeholder.markdown(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        else:
            st.warning("Please enter a query.")

    else:
        st.info("No documents in the knowledge base. Please upload a PDF in the sidebar.")

if __name__ == "__main__":
    st.session_state.num_chat_messages = 5
    main()