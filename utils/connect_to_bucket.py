from pathlib import Path
from oauth2client.service_account import ServiceAccountCredentials
from gcloud import storage

SERVICE_ACCOUNT_FILE = 'configs/conicle-ai.json'

credentials = ServiceAccountCredentials.from_json_keyfile_name(
    SERVICE_ACCOUNT_FILE)
storage_client = storage.Client(credentials=credentials, project='conicle-ai')
bucket = storage_client.get_bucket('conicle-ai-conicle-x-audio')

######### TRANSCRIPT BUCKET #########
TRANSCRIPT_BUCKET_NAME = 'conicle-ai-conicle-x-transcripts'
prefix = 'data/transcripts/'
dl_dir = 'transcripts/'


def download_from_bucket():
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        'configs/conicle-ai.json')
    storage_client = storage.Client(credentials=credentials, project='conicle-ai')
    bucket = storage_client.get_bucket(TRANSCRIPT_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    for blob in blobs:

        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        directory = directory.replace(prefix, dl_dir)
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name.replace(prefix, dl_dir))