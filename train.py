"""Train the model."""
import glob
import os
import math

import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import load_model

from model.flowchroma_network import FlowChroma
from model.fusion_layer import FusionLayer

from dataset.utils.shared import frames_per_video, default_nn_input_width, default_nn_input_height, dir_lab_records, \
    dir_resnet_csv
from dataset.data_generator import DataGenerator

import argparse

parser = argparse.ArgumentParser(description='Resize videos from')

parser.add_argument('-r', '--resnet-records',
                    type=str,
                    metavar='FILE',
                    dest='resnet_path',
                    default=dir_resnet_csv,
                    help='directory to resnet records')
parser.add_argument('-l', '--lab-records',
                    type=str,
                    metavar='FILE',
                    dest='lab_path',
                    default=dir_lab_records,
                    help='directory to lab records')
parser.add_argument('-s', '--split-ratio',
                    type=float,
                    default=0.1,
                    dest='val_split_ratio',
                    help='validation split ratio')
parser.add_argument('-t', '--train-batch-size',
                    type=int,
                    dest='train_batch_size',
                    default=4,
                    help='batch size of training set')
parser.add_argument('-v', '--val-batch-size',
                    type=int,
                    dest='val_batch_size',
                    default=4,
                    help='batch size of validation set')
parser.add_argument('-e', '--epochs',
                    type=int,
                    dest='n_epochs_to_train',
                    default=10,
                    help='number of epochs to train')
parser.add_argument('-c', '--ckpt-period',
                    type=int,
                    dest='ckpt_period',
                    default=2,
                    help='checkpoint period')

args = parser.parse_args()

resnet_path = args.resnet_path
lab_path = args.lab_path
val_split_ratio = args.val_split_ratio
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
n_epochs_to_train = args.n_epochs_to_train
ckpt_period = args.ckpt_period

initial_epoch = 0
ckpts = glob.glob("checkpoints/*.hdf5")
if len(ckpts) != 0:
    # there are ckpts
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print("loading from checkpoint:", latest_ckpt)
    initial_epoch = int(latest_ckpt[latest_ckpt.find("-epoch-") + len("-epoch-"):latest_ckpt.rfind("-train_acc-")])
    model = load_model(latest_ckpt, custom_objects={'FusionLayer': FusionLayer})

else:
    # no ckpts
    time_steps, h, w = frames_per_video, default_nn_input_height, default_nn_input_width

    enc_input = Input(shape=(time_steps, h, w, 1), name='encoder_input')
    incep_out = Input(shape=(time_steps, 1001), name='inception_input')

    model = FlowChroma([enc_input, incep_out]).build()
    # generate_model_summaries(model)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    os.makedirs("checkpoints", exist_ok=True)

n_lab_records = len(glob.glob('{0}/*.npy'.format(dir_lab_records)))
n_resnet_records = len(glob.glob('{0}/*.npy'.format(dir_resnet_csv)))

assert n_lab_records == n_resnet_records

val_split = math.floor(n_lab_records * val_split_ratio)

dataset = {
    "train": ['{0:05}'.format(i) for i in range(val_split)],
    "validation": ['{0:05}'.format(i) for i in range(val_split, n_lab_records)]
}

# generators
training_generator = DataGenerator(resnet_path=resnet_path,
                                   lab_path=lab_path,
                                   file_ids=dataset['train'],
                                   batch_size=train_batch_size)

validation_generator = DataGenerator(resnet_path=resnet_path,
                                     lab_path=lab_path,
                                     file_ids=dataset['validation'],
                                     batch_size=val_batch_size)

os.makedirs("checkpoints", exist_ok=True)
file_path = "checkpoints/flowchroma-epoch-{epoch:05d}-train_acc-{acc:.4f}-val_acc-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_path,
                             monitor=['acc', 'val_acc'],
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=ckpt_period)

if n_epochs_to_train <= initial_epoch:
    n_epochs_to_train += initial_epoch

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=n_epochs_to_train,
                    initial_epoch=initial_epoch,
                    callbacks=[checkpoint],
                    workers=6)
K.clear_session()
