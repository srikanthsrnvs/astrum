#!/bin/bash

tensorflow_model_server --model_base_path=$1 --rest_api_port=$2 --model_name=$3