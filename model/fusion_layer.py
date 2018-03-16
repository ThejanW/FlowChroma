from keras import backend as K
from keras.engine import Layer


class FusionLayer(Layer):
    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        enc_out, incep_out, rnn_out = inputs
        enc_out_shape = enc_out.shape.as_list()
        batch_size, time_steps, h, w, _ = map(lambda x: -1 if x is None else x, enc_out_shape)

        def _repeat(emb):
            # keep batch_size, time_steps axes unchanged
            # while replicating features h*w times
            emb_rep = K.tile(emb, [1, 1, h * w])
            return K.reshape(emb_rep, (batch_size, time_steps, h, w, emb.shape[2]))

        incep_rep = _repeat(incep_out)
        rnn_rep = _repeat(rnn_out)
        return K.concatenate([enc_out, incep_rep, rnn_rep], axis=4)

    def compute_output_shape(self, input_shapes):
        enc_out_shape, incep_out_shape, rnn_out_shape = input_shapes

        # Must have 3 tensors as input
        assert input_shapes and len(input_shapes) == 3

        # Batch size of the three tensors must match
        assert enc_out_shape[0] == incep_out_shape[0] == rnn_out_shape[0]

        # Number of time steps of the three tensors must match
        assert enc_out_shape[1] == incep_out_shape[1] == rnn_out_shape[1]

        # batch_size, time_steps, h, w, enc_out_depth = map(lambda x: -1 if x == None else x, enc_out_shape)
        batch_size, time_steps, h, w, enc_out_depth = enc_out_shape
        final_depth = enc_out_depth + incep_out_shape[2] + rnn_out_shape[2]
        return batch_size, time_steps, h, w, final_depth
