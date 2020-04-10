import os
import uuid
from zipfile import ZipFile

import firebase_admin
from firebase_admin import credentials, db, storage


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

    def save_model(self, job_id):
        blob = self.instance.bucket.blob('models/'+job_id)
        blob.upload_from_filename(job_id+'.h5')
        print("Model uploaded")
        return blob.public_url

    def save_tb_logs(self, job_id):
        blob = self.instance.bucket.blob('tb_logs/'+job_id)
        zip_name = '{}_tensorboard.zip'.format(job_id)
        with ZipFile(zip_name, 'w') as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(job_id+'_logs/'):
                for filename in filenames:
                    # create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                    # Add file to zip
                    zipObj.write(filePath)

            # close the Zip File
            zipObj.close()
            blob.upload_from_filename(zip_name)
        print("TB logs uploaded")
        return blob.public_url

    def save_logs(self, job_id):
        blob = self.instance.bucket.blob('logs/'+job_id)
        blob.upload_from_filename(job_id+'_output.txt')
        print("Logs uploaded")
        return blob.public_url

    def get_job_data(self, job_id):
        job_data = self.instance.db.reference('/jobs/'+job_id).get()
        dataset_id = job_data.get('dataset', "")
        dataset_data = self.instance.db.reference(
            '/datasets/'+dataset_id).get()
        urls = []

        for child in dataset_data['child_datasets']:
            urls.append(child['link'])

        return job_data, urls

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

    def pop_prediction(self, job):
        self.instance.db.reference('/jobs/'+job+'/status').set(3)
        builders_enqueued = self.get_prediction_builders_enqeued()
        builders_enqueued.remove(job)
        self.instance.db.reference('/prediction_queue').set(builders_enqueued)

    def get_jobs_enqeued(self):
        jobs = self.instance.db.reference('/job_queue').get()
        return jobs

    def get_prediction_builders_enqeued(self):
        prediction_builder_jobs = self.instance.db.reference(
            '/prediction_queue').get()
        return prediction_builder_jobs
