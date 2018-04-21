"""
Given the L channel of a sequence of Lab frames (range [-1, +1]), output a prediction over
the related sequence of a and b channels in the range [-1, 1].
In the neck of the conv-deconv network use the features from a feature extractor
(e.g. Inception) and the RNN to fuse them with the conv output.
"""

from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, LSTM, TimeDistributed, UpSampling2D
from keras.models import Model
from keras.utils import plot_model

from model.fusion_layer import FusionLayer


class FlowChroma:
    def __init__(self, inputs):
        # provide encoder input and inception ResNet's output as model inputs
        self.enc_input, self.incep_out = inputs

    @staticmethod
    def _encoder(encoder_input):
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2), name='encoder_conv1')(
            encoder_input)
        x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'), name='encoder_conv2')(x)
        x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2), name='encoder_conv3')(x)
        x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'), name='encoder_conv4')(x)
        x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2), name='encoder_conv5')(x)
        x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='encoder_conv6')(x)
        x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='encoder_conv7')(x)
        x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'), name='encoder_conv8')(x)
        return x

    @staticmethod
    def _rnn(rnn_input):
        x = LSTM(256, return_sequences=True, name='rnn_lstm1')(rnn_input)
        x = LSTM(256, return_sequences=True, name='rnn_lstm2')(x)
        x = TimeDistributed(Dense(256, activation='relu'), name='rnn_dense1')(x)
        return x

    @staticmethod
    def _decoder(decoder_input):
        x = TimeDistributed(Conv2D(256, (1, 1), activation='relu'))(decoder_input)
        x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'), name='decoder_conv1')(x)
        x = TimeDistributed(UpSampling2D((2, 2)), name='decoder_upsamp1')(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), name='decoder_conv2')(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), name='decoder_conv3')(x)
        x = TimeDistributed(UpSampling2D((2, 2)), name='decoder_upsamp2')(x)
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), name='decoder_conv4')(x)
        x = TimeDistributed(Conv2D(2, (3, 3), activation='tanh', padding='same'), name='decoder_conv5')(x)
        x = TimeDistributed(UpSampling2D((2, 2)), name='decoder_upsamp3')(x)
        return x

    def build(self):
        x = self._encoder(self.enc_input)
        x = TimeDistributed(GlobalAveragePooling2D())(x)
        x = self._rnn(x)
        x = Model(inputs=self.enc_input, outputs=x)

        rnn_out = x.get_layer(name='rnn_dense1').output
        enc_out = x.get_layer(name='encoder_conv8').output

        x = FusionLayer()([enc_out, self.incep_out, rnn_out])
        x = self._decoder(x)

        model = Model(inputs=[self.enc_input, self.incep_out], outputs=x)
        return model

    @staticmethod
    def generate_model_summaries(model):
        model.summary(line_length=150)
        plot_model(model, to_file='flowchroma.png')
