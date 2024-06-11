import lancedb
import vertexai
import glob
import yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB
from vertexai.generative_models import GenerativeModel, Part


def read_configs(location):

    with open(location) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config

def create_vector_database(category=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table(
        "my_table",
        data=[
            {
                "vector": embeddings.embed_query("Hello World"),
                "text": "Hello World",
                "id": "1",
            }
        ],
        mode="overwrite",
    )

    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    doc_list = []
    dl_dir = 'transcripts/'
    if category is not None:
        print("CATEGORY CASE")
        for file in glob.glob(dl_dir + f'{category}/' + "/*.txt"):
            with open(file) as f:
                doc_list.append(f.read())
    else:
        print("ALL CATEGORY CASE")
        for file in glob.glob(dl_dir + "/*.txt"):
            with open(file) as f:
                doc_list.append(f.read())

    text_splitter = CharacterTextSplitter(separator=',', chunk_size=100000, chunk_overlap=1000)
    documents = text_splitter.create_documents(doc_list)
    print('doclist', doc_list)
    vector_store = LanceDB.from_documents(documents, embeddings, connection=table)

    return vector_store


def get_conversational_chain(prompt, credentials, setting_location):
    vertexai.init(project='conicle-ai', credentials=credentials)
    system_instruction = read_configs(setting_location)['system_instruction']

    model = GenerativeModel(model_name="gemini-1.5-flash",
                            system_instruction=system_instruction)

    response = model.generate_content(
        [prompt]
    )

    print(response.text)
    return response.text