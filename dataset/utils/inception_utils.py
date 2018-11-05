from keras import Model
from keras.models import load_model
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import keras.backend as K
from dataset.utils.shared import checkpoint_url

K.clear_session()
x = load_model(checkpoint_url)
y = x.get_layer('predictions').output
model = Model(inputs=x.input, outputs=y)


def inception_resnet_v2_predict(images):
    images = images.astype(np.float32)
    predictions = model.predict(preprocess_input(images))
    return predictions
