import os
import streamlit as st
import base64
import google.generativeai as genai
from google.oauth2 import service_account
from utils.llm import create_vector_database, get_conversational_chain, parse_category
from utils.connect_to_bucket import download_from_bucket
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def ingest_data():
    download_from_bucket()
    vector_store = create_vector_database('configs/llm_settings.yaml')
    st.session_state['vector_store'] = vector_store


def user_input(user_question, category):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    print(category)
    vectordb = Chroma(persist_directory='vector_store', embedding_function=embeddings, collection_name=category)
    print(vectordb._collection.count())
    doc = vectordb.search(query=user_question, search_type='similarity', k=10)
    prompt = f"""Context:\n {doc}?\n Question: \n{user_question}\n"""
    response = get_conversational_chain(prompt, credentials, 'configs/llm_settings.yaml')
    return response


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]

    st.session_state['category'] = None



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

    st.sidebar.button('ห้ามกด', on_click=ingest_data)

    def get_category(category):
        st.session_state['category'] = category
        st.write(f"Selected category: {category}")
        print(category)

        return category

    if st.sidebar.button("คู่มือการใช้งาน"):
        st.sidebar.markdown("""
                            ## How to Use This Project (โปรเจคนี้ทำมาเพื่อการทดลอง experiment, fine-tuning system instruction เพื่อนำไปใช้ใน AI สืบไป)
                            0. เลือกหัวข้อด้านซ้ายมือ
                            1. แล้วคุย-ปรึกษา-ถามได้เลย (ถามได้แค่ในหัวข้อที่กด)
                            2. **Clear Chat History**: Use the clear chat history button to reset the chat. (Reset บทสนทนา)

                            ### Features
                            - **Category Selection**: Filter documents by category.
                            - **AI Chatbot**: Interact with the AI for guidance and support.
                            - **Clear Chat History**: Reset the chat for a new session.
                        """)
        
    st.sidebar.button('ConicleX Course IC Plain-Paper 1 Exam Preparation: Make It Easy with Mind Map ตอนที่ 2',
                      on_click=get_category, args=('Finance',))
    st.sidebar.button('ConicleX Course Mastering Prompt Engineering Design for ChatGPT AI Part 2',
                      on_click=get_category, args=('Data Science',))
    # st.sidebar.button('ConicleSpace-Grow (BETA)', on_click=get_category, args=('ConicleSpace-Grow',))
    # st.sidebar.button('Conicle Piece of Cake', on_click=get_category, args=('Piece_of_cake',))
    st.sidebar.button('ConicleX Course Cybersecurity Awareness', on_click=get_category, args=('course_123',))
    st.sidebar.button('ConicleX The Mindset Makeover', on_click=get_category, args=('course_124',))
    st.sidebar.button('ConicleX How to Increase Your Confidence', on_click=get_category, args=('course_125',))
    st.sidebar.button('ConicleX Piece of Cake Good Communication', on_click=get_category, args=('course_126',))
    st.sidebar.button('ConicleX Piece of Cake Happy Workplace', on_click=get_category, args=('course_127',))
    st.sidebar.button('ConicleX Piece of Cake ISO', on_click=get_category, args=('course_128',))
    st.sidebar.button('ConicleX Piece of Cake Strategic Thinking', on_click=get_category, args=('course_129',))
    st.sidebar.button('ConicleX Piece of Cake คู่มือผู้จัดการพันธุ์ใหม่', on_click=get_category, args=('course_130',))
    st.sidebar.button('ConicleX Piece of Cake เพิ่มยอดขาย', on_click=get_category, args=('course_131',))

    # Main content area for displaying chat messages
    # Load the image
    image_path = 'Sorc-Ai.png'
    img_base64 = get_image_as_base64(image_path)

    # Display the image next to the title
    st.markdown(
        f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" style="width:50px;height:50px;margin-right:10px;">
                <h1 style="display:inline;">Experiment Conicle AI</h1>
            </div>
            """,
        unsafe_allow_html=True
    )
    st.write("Version 1.2a")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if 'category' not in st.session_state:
        st.session_state['category'] = None
        st.session_state['vector_store'] = None


    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "เลือกหัวข้อด้านซ้ายมือก่อน ถึงค่อยเริ่มบทสนทนา"}]

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
                category = st.session_state['category']
                if category:
                    category = parse_category(st.session_state.get('category'))
                    response = user_input(user_question=prompt, category=category)
                else:
                    response = "เลือกหัวข้อที่คุณจะถามก่อน"
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