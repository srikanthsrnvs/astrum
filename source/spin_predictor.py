import base64
import json
import logging
import subprocess
import zipfile
from pathlib import Path

import firebase_admin
import requests
from firebase_admin import *
from flask import Flask, blueprints, jsonify, request
from flask_cors import CORS
from six.moves import http_client
from tensorflow.keras.preprocessing import image

from firebase import FirebaseHelper
from predictor import Predictor

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


BASEURL = 'https://api.astrum.ai'
spun_tf_servers = []
next_port = 5010


@app.route('/serve/<job_id>', methods=['POST'])
def serve(job_id):
    global BASEURL
    global spun_tf_servers
    global next_port

    # try:
    job = FirebaseHelper().get_job_data(job_id)

    models_path = str(Path.home())+'/models/'+job_id
    models_zip_path = models_path+'.zip'

    FirebaseHelper().get_file(job.serving_model, models_zip_path)
    with zipfile.ZipFile(models_zip_path, 'r') as zip_ref:
        zip_ref.extractall(models_path)
    os.remove(models_zip_path)

    port = str(next_port)
    next_port += 1

    try:
        spun_tf_servers.append(subprocess.Popen(["tensorflow_model_server "
                                                 "--model_base_path={} "
                                                 "--rest_api_port={} --model_name={}".format(models_path+'/ServingModel', port, job_id)],
                                                stdout=subprocess.DEVNULL,
                                                shell=True,
                                                preexec_fn=os.setsid))
    finally:
        prediction_url = 'http://localhost:{}/v1/models/{}:predict'.format(
            port, job_id)

        return jsonify({'status': 'success', 'url': prediction_url}), 200

    # except:
    #     return jsonify({'status': 'error', 'reason': "serving error"}), 400


@app.route('/predict/<job_id>', methods=['POST'])
def predict(job_id):
    global BASEURL
    global spun_tf_servers
    global next_port

    job = FirebaseHelper().get_job_data(job_id)

    if job.type == 'image_classification' or 'object_detection':

        img = request.files.get('image', "")
        if not img:
            return jsonify({'error': "No image was provided"}), 400

        img = image.img_to_array(image.load_img(img, target_size=(299, 299))) / 255.

        img = img.astype('float16')

        # Creating payload for TensorFlow serving request
        payload = {
            "instances": [{'input_image': img.tolist()}]
        }

        # Making POST request
        r = requests.post(job.prediction_url, json=payload)

        # Decoding results from TensorFlow Serving server
        pred = json.loads(r.content.decode('utf-8'))
        print(pred)
        # Returning JSON response to the frontend
        return jsonify(pred), 200


@app.errorhandler(http_client.INTERNAL_SERVER_ERROR)
def unexpected_error(e):
    """Handle exceptions by returning swagger-compliant json."""
    logging.exception('An error occured while processing the request.')
    response = jsonify({
        'code': http_client.INTERNAL_SERVER_ERROR,
        'message': 'Exception: {}'.format(e)})
    response.status_code = http_client.INTERNAL_SERVER_ERROR
    return response


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    try:
        app.run(host='127.0.0.1', port=8080, debug=True)
    except KeyboardInterrupt:
        print('Shutting down all servers...')
        for server in spun_tf_servers:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        print('Servers successfully shutdown!')
