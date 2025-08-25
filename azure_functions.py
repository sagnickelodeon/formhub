import os
import json

from azure.storage.blob import BlobServiceClient
from io import StringIO

def get_blob_object():
    service = BlobServiceClient(account_url=os.getenv("AZURE_ACCOUNT_URL"), credential=os.getenv("AZURE_API_KEY"))

    container = service.get_container_client(container=os.getenv("AZURE_CONTAINER_NAME"))
    return container

# Function to read a CSV file from Azure Blob Storage
def read_file_from_blob(blob_name, container) -> StringIO:
    blob_client = container.get_blob_client(blob=blob_name)
    downloaded_blob = blob_client.download_blob().readall()
    return StringIO(downloaded_blob.decode('utf-8'))  # Return an in-memory file-like object

# Function to write a CSV file to Azure Blob Storage
def write_file_to_blob(blob_name, data, container):
    blob_client = container.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data, overwrite=True)