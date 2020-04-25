import os
import shutil
import uuid
from zipfile import ZipFile

import firebase_admin
from firebase_admin import credentials, db, storage

from job import Job


class FirebaseHelper:

    class __FirebaseHelper:

        def __init__(self):
            primary_bucket = 'astrumdashboard.appspot.com'
            cred = credentials.Certificate('firebase.json')
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'astrumdashboard.appspot.com',
                "databaseURL": "https://astrumdashboard.firebaseio.com/"
            })
            self.db = db
            self.bucket = storage.bucket()

    instance = None

    def __init__(self):
        if not FirebaseHelper.instance:
            FirebaseHelper.instance = FirebaseHelper.__FirebaseHelper()

    def save_model(self, job_id, path):
        blob = self.instance.bucket.blob('models/'+job_id)
        blob.upload_from_filename(path)
        print("Model uploaded to ", blob.public_url)
        return blob.public_url

    def save_tb_logs(self, job_id, path):
        blob = self.instance.bucket.blob('tensorboard/'+job_id)
        blob.upload_from_filename(path)
        print("TB logs uploaded to ", blob.public_url)
        return blob.public_url

    def save_serving_model(self, job_id, path):
        blob = self.instance.bucket.blob('serving_models/'+job_id)
        blob.upload_from_filename(path)
        print("Serving model uploaded to ", blob.public_url)
        return blob.public_url

    def save_logs(self, job_id, path):
        blob = self.instance.bucket.blob('logs/'+job_id)
        blob.upload_from_filename(path)
        print("Logs uploaded to ", blob.public_url)
        return blob.public_url

    def get_job_data(self, job_id):
        job_data = self.instance.db.reference('/jobs/'+job_id).get()
        job_data['job_id'] = job_id
        dataset_id = job_data.get('dataset', "")
        dataset_data = self.instance.db.reference(
            '/datasets/'+dataset_id).get()
        job = Job(dataset_data, job_data)
        return job

    def get_file(self, link, destination_file_name):
        # bucket_name = "your-bucket-name"
        # source_blob_name = "storage-object-name"
        # destination_file_name = "local/path/to/file"

        file_dir = link.split('/')[-2]
        file_name = link.split('/')[-1]
        blob = self.instance.bucket.blob('{}/{}'.format(file_dir, file_name))
        blob.download_to_filename(destination_file_name)

    def pop_job(self, job):
        self.instance.db.reference('/jobs/'+job+'/status').set(1)
        jobs = self.get_jobs_enqeued()
        jobs.remove(job)
        self.instance.db.reference('/job_queue').set(jobs)

    def get_jobs_enqeued(self):
        jobs = self.instance.db.reference('/job_queue').get()
        return jobs
