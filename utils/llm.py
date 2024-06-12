import lancedb
import vertexai
import glob
import yaml
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB
from langchain_chroma import Chroma
from vertexai.generative_models import GenerativeModel, Part


def read_configs(location):

    with open(location) as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def create_vector_database(category=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    # db = lancedb.connect("/tmp/lancedb")

    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    doc_list = []
    dl_dir = 'transcripts/'

    if category is not None:
        print("CATEGORY CASE")
        path = dl_dir + f'{category}/' + '**/*.txt'
        for file in glob.glob(path, recursive=True):
            with open(file) as f:
                temp = f.read()
                doc_list.append(temp)

    else:
        print("ALL CATEGORY CASE")
        path = dl_dir + '**/*.txt'
        for file in glob.glob(path, recursive=True):
            with open(file) as f:
                temp = f.read()
                doc_list.append(temp)

    text_splitter = CharacterTextSplitter(separator=',', chunk_size=100000, chunk_overlap=1000)
    documents = text_splitter.create_documents(doc_list)

    vector_store = Chroma.from_documents(documents, embeddings)

    return vector_store


def get_conversational_chain(prompt, credentials, setting_location):
    vertexai.init(project='conicle-ai', credentials=credentials)
    system_instruction = read_configs(setting_location)['system_instruction']

    model = GenerativeModel(model_name="gemini-1.5-flash",
                            system_instruction=system_instruction)

    response = model.generate_content(
        [prompt]
    )

    try:
        result = response.text
    except:
        result = response.candidates[0].content.parts[0]._raw_part.text

    return result
