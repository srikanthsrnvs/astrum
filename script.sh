#!/bin/bash

sudo apt-get update
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda=10.0.130-1
sudo apt-get -y install python3-pip
sudo apt-get -y install python-virtualenv
virtualenv -p /usr/bin/python3 virtualenvironment/astrum

gsutil cp -r gs://astrumdashboard.appspot.com/cudnn.tgz ~
tar -xvzf cudnn.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

git clone https://github.com/srikanthsrnvs/astrum.git
PASSWORD: c8b755060cdc9feed578c0ec0b06886f641f3aab

sudo mkdir models
cd virtualenvironment/astrum/bin
source activate
cd
cd astrum/source
pip3 install -r requirements.txt

echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server

sudo /virtualenv/astrum/bin/python3 spin_predictor.py
