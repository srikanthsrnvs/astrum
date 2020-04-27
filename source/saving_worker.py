import os
import shutil
from pathlib import Path

import requests


BASE_URL = 'https://api.astrum.ai'


class SavingWorker:

    def __init__(self, firebase_helper):
        self.path = Path(str(Path.home())+'/JobFiles')
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = str(self.path)
        self.firebase_helper = firebase_helper

    def save_output(self, data):
        job = data.get('job', '')
        label_map = data.get('label_map', '')
        stats = data.get('stats', '')
        self.job = job

        self.__make_archive(self.path+'/'+job.id+'/ServingModel', self.path+'/'+job.id+'/ServingModel.zip')
        self.__make_archive(self.path+'/'+job.id+'/Tensorboard', self.path+'/'+job.id+'/Tensorboard.zip')

        saved_serving_model_location = self.firebase_helper.save_serving_model(
            job.id, self.path+'/'+job.id+'/ServingModel.zip')
        saved_model_location = self.firebase_helper.save_model(
            job.id, self.path+'/'+job.id+'/model.h5')
        saved_logs_location = self.firebase_helper.save_logs(
            job.id, self.path+'/'+job.id+'/log.txt')
        saved_tb_logs_location = self.firebase_helper.save_tb_logs(
            self.job.id, self.path+'/'+job.id+'/Tensorboard.zip')
        self.__notify_backend_for_completion({
            'model': saved_model_location,
            'logs': saved_logs_location,
            'serving_model': saved_serving_model_location,
            'tb_logs': saved_tb_logs_location,
            'label_map': label_map,
            'stats': stats,
            'status': 2
        })
        self.__create_prediction_endpoint()

    def __create_prediction_endpoint(self):
        response = requests.post(
            'http://predict.astrum.ai/serve/'+self.job.id,
        )
        if response.status_code == 200:
            self.prediction_url = response.json().get('url')
            self.__notify_backend_for_completion({
                'status': 3,
                'prediction_url': self.prediction_url
            })
        else:
            print("Error when serving the model")
            # TODO: handle the error if serving failed
        self.__cleanup()

    def __notify_backend_for_completion(self, data):
        requests.put(
            BASE_URL+'/jobs/'+self.job.id,
            json=data
        )

    def __cleanup(self):
        shutil.rmtree(self.path+'/'+self.job.id)


    def __make_archive(self, source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)
