import tensorflow as tf
from tensorflow.python import keras
import random
import numpy as np
from custom_lenet import CustomLeNet
import os
from PIL import Image


class GenericBuilder:

    def __init__(self, type, input_size, output_classes):
        self.type = type
        self.input_size = input_size
        self.output_classes = output_classes

        if self.type == 'structured_classification':
            pass
        elif self.type == 'structured_prediction':
            pass
        elif self.type == 'image_classification':
            self.model = self.build_image_classifier()

    def build_image_classifier(self):
        model = CustomLeNet(self.input_shape, self.output_classes)
        return model

    def build_data_predictor(self):
        pass

    def prepare_image_classifier_dataset(self, urls):
        os.makedirs("datasets")
        total_img_count = 0
        cumalative_img_height = 0
        cumalative_img_width = 0
        imgs = []
        classes = []

        for url in urls:
            r = requests.get(url)
            f = io.BytesIO(r.content)
            z = zipfile.ZipFile(f)
            z.extractall()
            filename = z.filelist[0].filename.strip('/')
            path = 'datasets/'+filename
            os.rename(filename, path)
            classes.append(filename)
            for file in os.listdir(path):
                img = Image.open(os.path.join(path, file))
                img_name = file.split('.')[0:-1]
                img_extension = file.split('.')[-1]
                imgs.append({'image': img, 'class': filename,
                             'name': img_name, 'extension': img_extension})
                total_img_count += 1
                img_height, img_width = img.size
                cumalative_img_height += img_height
                cumalative_img_width += img_width

        img_size = max(cumalative_img_height/total_img_count,
                       cumalative_img_width/total_img_count)

        # Save all images by splitting into /test & /train
        for cls in classes:
            os.makedirs('datasets/train/'+cls)
            os.makedirs('datasets/test/'+cls)
            # Reshape all images
            for im in imgs:
                img = im['image'].resize((img_size, img_size))
                if im['class'] == cls:
                    # TODO: Need to split into test&train set, also need to augment
                    img.save('datasets/train/{}/{}'.format(cls,
                                                           im['name']), im['extension'])
