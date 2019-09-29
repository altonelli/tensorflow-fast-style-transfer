# for colab
# from google.colab import auth
from private_constants import PrivateConstants

import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.cloud import storage


class GCS:
    def list_files_from_gcs(bucket_name, path):
        gcs_service = storage.Client(PrivateConstants.PROJECT_NAME)
        bucket = gcs_service.bucket(bucket_name)
        file_names = []
        blobs = bucket.list_blobs(prefix=path)
        for blob in blobs:
            file_names.append(blob.name)
        return file_names

    def download_files_from_gcs(bucket_name, path, to_dir):
        gcs_service = storage.Client(PrivateConstants.PROJECT_NAME)
        bucket = gcs_service.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=path)
        for blob in blobs:
            file_name = blob.name.rsplit('/', 1)[-1]
            if file_name == "":
                continue
            blob.download_to_filename(os.path.join(to_dir, file_name))
        print("Download to {}/{} complete".format(bucket_name, path))
        return True

    def upload_to_gcs(bucket_name, from_file, to_file):
        gcs_service = storage.Client(PrivateConstants.PROJECT_NAME)
        bucket = gcs_service.bucket(bucket_name)
        blob = bucket.blob(to_file)
        blob.upload_from_filename(from_file)
        print("Upload to {}/{} complete".format(bucket_name, to_file))
        return True

    def download_file_from_gcs(bucket_name, from_file, to_file):
        gcs_service = storage.Client(PrivateConstants.PROJECT_NAME)
        bucket = gcs_service.bucket(bucket_name)
        blob = bucket.blob(from_file)
        blob.download_to_filename(to_file)
        print("Download from {}/{} complete".format(bucket_name, from_file))
        return True

    # DEPRECATED
    def get_from_gcs(bucket_name, from_file, to_file):
        # Create the service client.
        gcs_service = build('storage', 'v1')

        with open(to_file, 'wb') as f:
            request = gcs_service.objects().get_media(bucket=bucket_name,
                                                      object=from_file)
            media = MediaIoBaseDownload(f, request)

            done = False
            while not done:
                # _ is a placeholder for a progress object that we ignore.
                # (Our file is small, so we skip reporting progress.)
                _, done = media.next_chunk()

        print('Download complete')

    # DEPRECATED
    def store_in_gcs(bucket_name, from_file, to_file):
        # Create the service client.
        gcs_service = build('storage', 'v1')

        #   name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))

        with open(to_file, 'wb') as f:
            request = gcs_service.objects().get_media(bucket=bucket_name,
                                                      object=from_file)
            media = MediaIoBaseDownload(f, request)

            done = False
            while not done:
                # _ is a placeholder for a progress object that we ignore.
                # (Our file is small, so we skip reporting progress.)
                _, done = media.next_chunk()

        print('Upload complete')
