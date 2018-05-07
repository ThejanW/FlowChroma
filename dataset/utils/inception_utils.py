import tensorflow as tf
from keras import Model
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np


def inception_resnet_v2_predict(images):
    images = images.astype(np.float32)
    x = InceptionResNetV2(weights='imagenet', include_top=True)
    y = x.layers[-2].output
    model = Model(inputs=x.input, outputs=y)
    predictions = model.predict(preprocess_input(images))
    return predictions
