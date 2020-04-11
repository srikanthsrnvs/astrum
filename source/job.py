import os
import zipfile
import requests
import io
from PIL import Image


class Job:

    def __init__(self, data_snapshot, job_data_snapshot):
        self.label_map = job_data_snapshot['label_map']
        self.label_map = dict((v, k) for k, v in self.label_map.items())
        self.model = job_data_snapshot['model']
        self.type = data_snapshot.get('dataset_type', '')
        self.file_data = data_snapshot.get('file', '')
        self.id = job_data_snapshot['job_id']
        self.download_link = self.file_data.get('link', "")
        self.created_by = data_snapshot['uploaded_by']

    def set_filename(self, filename):
        self.filename = filename