import tensorflow as tf
from keras import Model
from keras.applications import InceptionResNetV2


def prepare_image_for_inception(input_tensor):
    """
    Pre-processes an image tensor ``(int8, range [0, 255])``
    to be fed into inception ``(float32, range [-1, +1])``

    :param input_tensor:
    :return:
    """
    res = tf.cast(input_tensor, dtype=tf.float32)
    res = 2 * res / 255 - 1
    res = tf.reshape(res, [-1, 299, 299, 3])
    return res


def inception_resnet_v2_predict(images):
    x = InceptionResNetV2(weights='imagenet', include_top=True)
    y = x.layers[-2].output
    model = Model(inputs=x.input, outputs=y)
    predictions = model.predict(images)
    return predictions
