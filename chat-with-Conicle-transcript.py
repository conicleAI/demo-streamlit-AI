import os
import streamlit as st
import base64
import google.generativeai as genai
from google.oauth2 import service_account
from utils.llm import create_vector_database, get_conversational_chain, parse_category
from utils.connect_to_bucket import download_from_bucket
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers import pipeline

# Configure credentials
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Constants
QUESTION_LIMIT = 3


# Helper functions
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def intent_recognize(prompt):
    INTENTS = ["gratitude", "summary"]
    #oracle = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    #result = oracle(prompt, candidate_labels=INTENTS)

    #if(result["score"][0] > 0.5):
    #    with st.chat_message("user"):

    #        st.write(result["labels"][0])


    return None

def ingest_data():
    download_from_bucket()
    vector_store = create_vector_database('configs/llm_settings.yaml')
    st.session_state['vector_store'] = vector_store


def user_input(user_question, category):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory='vector_store', embedding_function=embeddings, collection_name=category)
    doc = vectordb.search(query=user_question, search_type='similarity', k=10)
    prompt = f"Context:\n {doc}?\n Question: \n{user_question}\n"
    response = get_conversational_chain(prompt, credentials, 'configs/llm_settings.yaml')
    return response


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]
    st.session_state['conversation_history'] = []  # Clear the conversation history
    st.session_state['category'] = None
    st.session_state['question_count'] = 0  # Reset the question count


# Main application
def main():
    st.set_page_config(page_title="Conicle Dos Chatbot", page_icon="🤖")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        st.button('ห้ามกด', on_click=ingest_data)

        def get_category(category):
            st.session_state['category'] = category
            st.write(f"Selected category: {category}")
            return category

        if st.button("คู่มือการใช้งาน"):
            st.markdown("""
                Scenario: จำลองว่าคุณเป็นผู้ใช้งาน กำลังเรียน คอร์สที่เลือก แล้วระหว่างเรียน หรือหลังเรียน จะพิมพ์ถามอะไร อาจารย์

                ## (โปรเจคนี้ทำมาเพื่อการทดลอง experiment, fine-tuning เพื่อนำไปปรับปรุง Conicle AI)
                0. เลือกหัวข้อด้านซ้ายมือ
                1. แล้วคุย-ปรึกษา-ถามได้เลย (ถามได้แค่ในหัวข้อที่กด)
                2. **Clear Chat History**: Use the clear chat history button to reset the chat. (Reset บทสนทนา)
                ### Features
                - **Category Selection**: Filter documents by category.
                - **AI Chatbot**: Interact with the AI for guidance and support.
                - **Clear Chat History**: Reset the chat for a new session.
            """)

        # Category buttons
        categories = {
            'Finance': 'ConicleX Course IC Plain-Paper 1 Exam Preparation: Make It Easy with Mind Map ตอนที่ 2',
            'Data Science': 'ConicleX Course Mastering Prompt Engineering Design for ChatGPT AI Part 2',
            'course_123': 'ConicleX Course Cybersecurity Awareness',
            'course_124': 'ConicleX The Mindset Makeover',
            'course_125': 'ConicleX How to Increase Your Confidence',
            'course_126': 'ConicleX Piece of Cake Good Communication',
            'course_127': 'ConicleX Piece of Cake Happy Workplace',
            'course_128': 'ConicleX Piece of Cake ISO',
            'course_129': 'ConicleX Piece of Cake Strategic Thinking',
            'course_130': 'ConicleX Piece of Cake คู่มือผู้จัดการพันธุ์ใหม่',
            'course_131': 'ConicleX Piece of Cake เพิ่มยอดขาย'
        }

        for key, value in categories.items():
            st.button(value, on_click=get_category, args=(key,))

        st.button('Clear Chat History', on_click=clear_chat_history)

    # Main content area
    image_path = 'Sorc-Ai.png'
    img_base64 = get_image_as_base64(image_path)

    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{img_base64}" style="width:50px;height:50px;margin-right:10px;">
            <h1 style="display:inline;">Coniverse AI</h1>
        </div>
    """, unsafe_allow_html=True)

    st.write("Version 3.2b")

    # Initialize session state
    if 'category' not in st.session_state:
        st.session_state['category'] = None
        st.session_state['vector_store'] = None

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "เลือกหัวข้อด้านซ้ายมือก่อน ถึงค่อยเริ่มบทสนทนา"}
        ]

    if "conversation_history" not in st.session_state.keys():
        st.session_state['conversation_history'] = []

    if "question_count" not in st.session_state.keys():
        st.session_state['question_count'] = 0

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input():
        if st.session_state['question_count'] < QUESTION_LIMIT:
            intent_recognize(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        category = st.session_state['category']
                        if category:
                            category = parse_category(st.session_state.get('category'))
                            st.session_state['conversation_history'].append(prompt)
                            conversation_context = " ".join(st.session_state['conversation_history'])
                            response = user_input(user_question=f"{conversation_context}\n{prompt}", category=category)
                        else:
                            response = "เลือกหัวข้อที่คุณจะถามก่อน"
                        placeholder = st.empty()
                        full_response = ''
                        for item in response:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)

                if response:
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)
                    st.session_state['conversation_history'].append(full_response)
                    st.session_state['question_count'] += 1
        else:
            st.session_state.messages.append({"role": "assistant", "content": "End of session. Subscribe for more."})
            with st.chat_message("assistant"):
                st.write("End of session. Subscribe for more.")


if __name__ == "__main__":
    main()
