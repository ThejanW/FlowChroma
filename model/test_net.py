import numpy as np
from keras.layers import Input

from model.flowchroma_network import FlowChroma

batch_size, time_steps, h, w = 4, 10, 240, 320

enc_input = Input(shape=(time_steps, h, w, 1), name='encoder_input')
incep_out = Input(shape=(time_steps, 1000), name='inception_input')

model = FlowChroma([enc_input, incep_out]).build()
# generate_model_summaries(model)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# generate dummy data
X = [np.random.random((batch_size, time_steps, h, w, 1)), np.random.random((batch_size, time_steps, 1000))]
Y = np.random.random((batch_size, time_steps, h, w, 2))

# before fitting
_rnn_w_1 = model.get_layer(name='rnn_lstm2').get_weights()

model.fit(X, Y, epochs=10, batch_size=2)

# after fitting
_rnn_w_2 = model.get_layer(name='rnn_lstm2').get_weights()

print("##############################################################################")
print(_rnn_w_1)
print("##############################################################################")
print(_rnn_w_2)
print("##############################################################################")
print(np.array_equal(_rnn_w_1, _rnn_w_2))
