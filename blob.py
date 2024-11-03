from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient
import os
import os.path
from dotenv import load_dotenv
load_dotenv()
account_name = 'apnawaqeel'
account_key = os.getenv('BLOB_KEY')
connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# blob_name = '1-co-operative-societies-act-1925-pdf.pdf'
# blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
# download_file_path = r"C:\Users\Nouma\OneDrive\Desktop\GenAi x Xavour\LangGraph\1-co-operative-societies-act-1925-pdf.pdf"
# with open(download_file_path, "wb") as download_file:
#     download_file.write(blob_client.download_blob().readall())
# print("local_file_path")


def download_Blob(pdf_name):
    container_name = 'pdfs'
    for i in range(len(pdf_name)):
        blob_name = pdf_name[i]
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        download_file_path = r"C:\Users\Nouma\OneDrive\Desktop\GenAi x Xavour\LangGraph\\" + pdf_name[i]
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    return "Done"
