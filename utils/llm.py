import os
import datetime
import lancedb
import vertexai
import glob
import yaml
import google.generativeai as genai
import streamlit as st
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB
from langchain_community.vectorstores.chroma import Chroma
from vertexai.generative_models import GenerativeModel, Part
from google.generativeai import caching
from google.generativeai import protos


def read_configs(location):

    with open(location) as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def parse_category(category_name):

    parse_name = category_name.lower()
    parse_name = parse_name.replace(" ", "_")
    return parse_name


def create_vector_database(location):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    # db = lancedb.connect("/tmp/lancedb")

    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    dl_dir = 'transcripts/'
    category_list = read_configs(location)['category_list']

    for category in category_list:
        doc_list = []
        print("CATEGORY CASE")
        path = dl_dir + f'{category}/' + '**/*.txt'
        print(path)
        for file in glob.glob(path, recursive=True):
            print(file)
            with open(file) as f:
                temp = f.read()
                doc_list.append(temp)
        text_splitter = CharacterTextSplitter(separator=',', chunk_size=100000, chunk_overlap=1000)
        documents = text_splitter.create_documents(doc_list)
        category = parse_category(category)
        ids = [str(i) for i in range(1, len(doc_list) + 1)]
        vector_store = Chroma.from_documents(documents, embeddings, ids=ids, collection_name=category, persist_directory='vector_store')
        vector_store.persist()


    return vector_store


def get_conversational_chain(context, prompt, credentials, setting_location):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    system_instruction = read_configs(setting_location)['system_instruction']

    context_path = "temp.txt"
    string_context = [context[i].page_content for i in range(len(context))]
    string_context = '<doc end>'.join(string_context)
    with open(context_path, "w") as text_file:
        text_file.write(string_context)
    contents = genai.upload_file(path=context_path)

    while contents.state.name == "PROCESSING":
        print('Waiting for contents to be processed.')
        contents = genai.get_file(contents.name)  

    try:
        cache = caching.CachedContent.create(
            model="models/gemini-1.5-flash-001",
            display_name="related contents",  # used to identify the cache
            system_instruction=system_instruction,
            contents=[contents],
            ttl=datetime.timedelta(minutes=5),
        )

        model = genai.GenerativeModel.from_cached_content(cached_content=cache, generation_config=generation_config)

        response = model.generate_content(
            [prompt]
    )

    except:
        prompt = prompt + string_context
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )

        response = model.generate_content(
            [prompt]
        )

    print(response)
    try:
        result = response.text
    except:
        result = response.candidates[0].content.parts[0].text

    return result
