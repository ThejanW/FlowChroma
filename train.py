"""Train the model."""

from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import load_model

from model.flowchroma_network import FlowChroma
from model.fusion_layer import FusionLayer

from dataset.utils.shared import frames_per_video, default_nn_input_width, default_nn_input_height
from dataset.data_generator import DataGenerator

time_steps, h, w = frames_per_video, default_nn_input_height, default_nn_input_width

enc_input = Input(shape=(time_steps, h, w, 1), name='encoder_input')
incep_out = Input(shape=(time_steps, 1001), name='inception_input')

model = FlowChroma([enc_input, incep_out]).build()
# generate_model_summaries(model)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

dataset = {
    "train": ['{0:05}'.format(i) for i in range(57)],
    "validation": ['{0:05}'.format(i) for i in range(57, 64)]
}

# generators
training_generator = DataGenerator(dataset['train'], batch_size=2)
validation_generator = DataGenerator(dataset['validation'], batch_size=2)

# checkpoint whenever an improvement happens in val accuracy
filepath = "flowchroma-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model = load_model('flowchroma-final.h5', custom_objects={'FusionLayer': FusionLayer})
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=4,
                    callbacks=callbacks_list,
                    workers=6)
# save final weights
model.save('flowchroma-final.h5')
