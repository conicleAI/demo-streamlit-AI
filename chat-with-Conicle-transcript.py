import os
import streamlit as st
import base64
import google.generativeai as genai
from google.oauth2 import service_account
from utils.llm import create_vector_database, get_conversational_chain
from utils.connect_to_bucket import download_from_bucket


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def ingest_data():
    download_from_bucket()


def initializing(category=None):

    vector_store = create_vector_database(category)

    return vector_store


def user_input(vector_store, user_question):

    doc = vector_store.similarity_search(user_question, k=4)
    prompt = f"""Context:\n {doc}?\n Question: \n{user_question}\n"""

    response = get_conversational_chain(prompt)
    return response


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]


def main():
    st.set_page_config(
        page_title="Conicle Dos Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        # txt_file = st.file_uploader(
        #     "Upload your file(s) and Click on the Submit & Process Button", accept_multiple_files=False)

    st.sidebar.button('Ingest Data', on_click=ingest_data)

    def get_category(category):
        st.session_state['category'] = category
        st.write(category)
        print(category)
        vector_store = initializing(category)
        st.session_state['vector_store'] = vector_store

        return category

    st.sidebar.button('Finance', on_click=get_category, args=('Finance',))
    st.sidebar.button('Data Science', on_click=get_category, args=('Data Science',))
    st.sidebar.button('ConicleSpace-Grow (BETA)', on_click=get_category, args=('ConicleSpace-Grow',))
    st.sidebar.button('Conicle Piece of Cake', on_click=get_category, args=('Piece_of_cake',))

    # Main content area for displaying chat messages
    # Load the image
    image_path = 'Sorc-Ai.png'
    img_base64 = get_image_as_base64(image_path)

    # Display the image next to the title
    st.markdown(
        f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" style="width:50px;height:50px;margin-right:10px;">
                <h1 style="display:inline;">Brae V8.1a</h1>
            </div>
            """,
        unsafe_allow_html=True
    )
    st.write("Conicle's Bare V8.1a")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if 'category' not in st.session_state:
        st.session_state['category'] = None

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "มีอะไรให้หนูช่วยไมค่ะ?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                response = user_input(vector_store=st.session_state['vector_store'], user_question=prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()