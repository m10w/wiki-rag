import os
import azure.storage.blob as azure_blob
import io
import logging
from dotenv import load_dotenv


def upload_to_blob(content, container_name, blob_name):

    # Azure Blob Storage setup
    try:
        load_dotenv()
        connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")

        # Upload to Blob
        blob_service_client = azure_blob.BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Convert content to a BytesIO stream
        content_stream = io.BytesIO(content.encode('utf-8'))

        # Upload directly to Blob Storage
        blob_client.upload_blob(content_stream, overwrite=True)
        logging.info(f"Uploaded {blob_name} to container {container_name}.")

    except Exception as e:
        logging.error(f"Failed to upload data to Azure Blob Storage: {e}")
