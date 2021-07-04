import os.path
from shutil import copyfile

import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

from src.sorters.abstract import AbstractSorter


def prepare_example(img_fp, label=None, target_size=(224, 224)):
    img_fp = img_fp.numpy().decode()
    return tf.constant(cv2.resize(cv2.cvtColor(cv2.imread(img_fp), cv2.COLOR_BGR2RGB), target_size), dtype=tf.uint8), tf.constant(label or '', dtype=tf.string)

def prepare_example_wrap(*args):
    return tf.py_function(prepare_example, inp=args, Tout=(tf.uint8, tf.string))


class Categorizer(AbstractSorter):
    def __init__(self, input_path: str, output_path: str, config_path: str = 'configs/categorizer.yml'):
        super().__init__(input_path, output_path, config_path)

    def process(self):
        backbone = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
        model = tf.keras.Model(inputs=backbone.inputs, outputs=backbone.layers[-2].output)
        clusterer = joblib.load(self.config['clusterer_path'])
        num_clusters = self.config['num_clusters']
        data_loader = tf.data.Dataset.list_files(os.path.join(self.input_path, '*'))

        imgs, features = [], []
        print('Processing image features...')
        for image, _ in tqdm(data_loader.map(prepare_example_wrap)):
            image, feats = image.numpy(), self._extract_features(model, image)
            imgs.append(image)
            features.append(feats[0])
        
        features = np.array(features)
        preds = clusterer.predict(features) 

        print('Clusterring images...')
        img_paths = np.array([fp.numpy().decode() for fp in data_loader])
        clusters = []
        for cluster_id in tqdm(range(num_clusters)):
            cluster_image_paths = img_paths[preds == cluster_id]
            if cluster_image_paths.shape[0] == 0:
                continue
            
            for fp in cluster_image_paths.tolist():
                fname = os.path.split(fp)[1]
                dst_dir = os.path.join(self.output_path, str(cluster_id))
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                copyfile(fp, os.path.join(dst_dir, fname))
            clusters.append(cluster_image_paths.tolist())
        print('Done.')
        return clusters

    @staticmethod
    def _extract_features(model, img):
        img = preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        return model.predict(img, use_multiprocessing=True)
