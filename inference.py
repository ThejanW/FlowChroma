import glob
import os
from os import listdir
import cv2
import numpy as np
from keras.models import load_model
from skimage import color

from dataset.utils.inception_utils import inception_resnet_v2_predict
from dataset.utils.resize import resize_pad_frame
from dataset.utils.shared import frames_per_video, default_nn_input_width, default_nn_input_height, resnet_input_height, resnet_input_width, dir_test, dir_test_results
from model import FusionLayer
import keras.backend as K

def get_video(file):
    '''
    Parameters
    ----------
    file - path to video file

    Returns
    -------
    frames - frames array of the video
    '''

    video = cv2.VideoCapture(file)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.asarray(frames)
    return frames


def get_lab_layer(frames):
    '''
        Parameters
        -----------
        frames - color/gray video frames with 3 chanels

        Returns
        -------
        (rgb2lab, gray2lab) - RGB frames converted to LAB, GRAY frames converted to LAB
    '''
    rgb2lab_frames = []
    gray2lab_frames = []

    for frame in frames:
        resized_frame = resize_pad_frame(frame, (default_nn_input_height, default_nn_input_width), equal_padding=True)

        rgb2lab_frame = color.rgb2lab(resized_frame)
        rgb2lab_frames.append(rgb2lab_frame)

        rgb2gray_frame = color.rgb2gray(resized_frame)

        #         Display Grayscale frame
        #         cv2.imshow('grey', rgb2gray_frame)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()

        gray2rgb_frame = color.gray2rgb(rgb2gray_frame)
        lab_frame = color.rgb2lab(gray2rgb_frame)
        gray2lab_frames.append(lab_frame)

    return np.asarray(rgb2lab_frames), np.asarray(gray2lab_frames)


def preprocess_frames(gray2lab_frames):
    '''
    Parameters
    ---------
    gray2lab_frames - LAB frames of Grayscale video

    Returns
    -------
    processed_l_layer - L Layer processed (L/50 - 1)
    '''
    processed = np.empty(gray2lab_frames.shape)

    processed[:, :, :, 0] = np.divide(gray2lab_frames[:, :, :, 0], 50) - 1  # data loss
    processed[:, :, :, 1] = np.divide(gray2lab_frames[:, :, :, 1], 128)
    processed[:, :, :, 2] = np.divide(gray2lab_frames[:, :, :, 2], 128)

    processed_l_layer = processed[:, :, :, np.newaxis, 0]

    return processed_l_layer


def get_resnet_records(frames):
    '''
    Parameters
    ----------
    frames - original frames without color conversion or resizing

    Details
    -------
    Implementation adopted from Deep Kolorization implementation

    Returns
    -------
    predictions - restnet predictions
    '''
    resnet_input = []
    for frame in frames:
        resized_frame = resize_pad_frame(frame, (resnet_input_height, resnet_input_width))
        gray_scale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
        gray_scale_frame_colored = cv2.cvtColor(gray_scale_frame, cv2.COLOR_GRAY2RGB)
        resnet_input.append(gray_scale_frame_colored)
    resnet_input = np.asarray(resnet_input)

    predictions = inception_resnet_v2_predict(resnet_input)
    return predictions


def getInputRange(frames_count, time_steps, current_frame):
    '''
    Deciding the moving window
    '''
    # this function should change according to our selection of
    frame_selection = []
    last_selection = current_frame
    for i in range(current_frame, current_frame - time_steps, -1):
        if (i < 0):
            frame_selection.append(last_selection)
        else:
            frame_selection.append(i)
            last_selection = i
    frame_selection = frame_selection[::-1]
    return frame_selection


def get_nn_input(l_layer, resnet_out):
    '''
    Define the flowchroma input
    '''
    frames_count = l_layer.shape[0]
    time_steps = frames_per_video
    X = []
    Y = []

    for i in range(frames_count):
        frame_index_selection = getInputRange(frames_count, time_steps, i)
        frame_selection = []
        resnet_selection = []
        for j in frame_index_selection:
            frame_selection.append(l_layer[j])
            resnet_selection.append(resnet_out[j])
        X.append(frame_selection)
        Y.append(resnet_selection)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return [X, Y]


def post_process_predictions(original_l_layers, predicted_AB_layers):
    '''
    Combine original L layer and predicted AB Layers
    '''
    time_steps = frames_per_video
    total_frames = original_l_layers.shape[0]
    predicted_frames = []
    for i in range(total_frames):
        l_layer = original_l_layers[i]
        a_layer = np.multiply(predicted_AB_layers[i, time_steps - 1, :, :, 0],
                              128)  # select the first frame outof three predictions
        b_layer = np.multiply(predicted_AB_layers[i, time_steps - 1, :, :, 1], 128)
        frame = np.empty((240, 320, 3))
        frame[:, :, 0] = l_layer
        frame[:, :, 1] = a_layer
        frame[:, :, 2] = b_layer
        # frame = color.lab2rgb(frame)
        predicted_frames.append(frame)
    return np.asarray(predicted_frames)


def save_output_video(frames, output_file):
    '''
    Save the output video
    '''
    fps = 20
    size = (320, 240)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)
    for frame in frames:
        final_out = color.lab2rgb(frame)
        final_out_write_video = final_out * 255  # color.lab2rgb results values in [0,1]
        final_out_write_video = final_out_write_video.astype(np.uint8)
        out.write(final_out_write_video)
    out.release()


def process_test_file(file):
    print('Processing file: ' + file)
    # Pre-processing
    frames = get_video(dir_test+'/'+file)
    (rgb2lab_frames, gray2lab_frames) = get_lab_layer(frames)
    processed_l_layer = preprocess_frames(gray2lab_frames)
    print('running resnet model')
    predictions = get_resnet_records(frames)
    print('Combining L laber and resnet out')
    X = get_nn_input(processed_l_layer, predictions)

    # Predicting
    predictions = []
    for i in range(X[0].shape[0]):
        predictions.append(model.predict([X[0][i:i + 1], X[1][i:i + 1]])[0])  # shape is (1, 3, 240, 320, 2)
    predictions = np.asarray(predictions)
    print("Flowchroma model predictions calculated")

    # Post processing
    frame_predictions = post_process_predictions(gray2lab_frames[:, :, :, 0], predictions)

    save_output_video(frame_predictions, dir_test_results+ '/' + file.split('.')[0] + '.avi')

K.clear_session()
ckpts = glob.glob("checkpoints/*.hdf5")
latest_ckpt = max(ckpts, key=os.path.getctime)
model = load_model(latest_ckpt, custom_objects={'FusionLayer': FusionLayer})
print("loading from checkpoint:", latest_ckpt)

files = listdir(dir_test)
for file in files:
    process_test_file(file)
