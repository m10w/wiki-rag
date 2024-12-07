import requests
import logging
import json
import os
import azure.functions as func
import azure.storage.blob as azure_blob
from azure.identity import DefaultAzureCredential
from unittest.mock import MagicMock


def main(mytimer: func.TimerRequest) -> None:
    logging.info('Wikipedia Data Extraction started.')

    # Wikipedia API URL
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": "Artificial Intelligence|Cloud Computing|Generative AI|RAG|Large Language Models",  # Example topics
        "explaintext": True,
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data: {response.status_code}")
        return

    data = response.json()

    # Azure Blob Storage setup
    try:
        connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        container_name = "wiki-data"
        blob_name = "data.json"

        # Upload to Blob
        blob_service_client = azure_blob.BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        json_data = json.dumps(data)
        blob_client.upload_blob(json_data, overwrite=True)
        logging.info("Data uploaded to Azure Blob Storage.")

    except Exception as e:
        logging.error(f"Failed to upload data to Azure Blob Storage: {e}")

if __name__ == "__main__":
    mock_timer = MagicMock()
    main(mock_timer)