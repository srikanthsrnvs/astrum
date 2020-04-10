import base64
import json
import logging

import firebase_admin
from firebase_admin import *
from flask import Flask, blueprints, jsonify, request
from flask_cors import CORS
from six.moves import http_client

from firebase import FirebaseHelper
from predictor import Predictor

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict/<job_id>', methods=['POST'])
def predict(job_id):

    job_data = FirebaseHelper().get_job_data(job_id)[0]
    job_type = job_data['type']
    if job_type == 'image_classification':

        img = request.files.get('image', "")
        if not img:
            return jsonify({'error': "No image was provided"}), 400

        label_map = dict((v, k) for k, v in job_data['label_map'].items())
        model_location = job_data['model']

        predictor = Predictor(model_location, job_id,
                              job_type, label_map=label_map)
        return predictor.predict(img)


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
    app.run(host='127.0.0.1', port=8080, debug=True)
