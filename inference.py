import glob
import os

import cv2
import numpy as np
from keras.models import load_model
from skimage import color

from dataset.utils.inception_utils import inception_resnet_v2_predict
from dataset.utils.resize import resize_pad_frame
from dataset.utils.shared import frames_per_video, default_nn_input_width, default_nn_input_height, resnet_input_height, resnet_input_width
from model import FusionLayer


def get_video(file):
    video = cv2.VideoCapture(file)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.asarray(frames)
    return frames


def get_l_layer(frames):
    lab_frames = []

    for frame in frames:
        resized_frame = resize_pad_frame(frame, (default_nn_input_height, default_nn_input_width), equal_padding=True)
        # Convert to grayscale and rgb and take the L layer ?
        gray_scale_frame = color.rgb2gray(resized_frame)
        gray_scale_colored_frame = color.gray2rgb(gray_scale_frame)
        lab_frame = color.rgb2lab(gray_scale_colored_frame)
        lab_frames.append(lab_frame)

    lab_frames = np.asarray(lab_frames)
    original_l_layers = np.copy(lab_frames[:, :, :, 0])
    lab_frames[:, :, :, 0] = np.divide(lab_frames[:, :, :, 0], 50) - 1  # data loss
    return original_l_layers, lab_frames[:, :, :, np.newaxis, 0]


def get_resnet_records(frames):
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
    :param frames_count:
    :param time_steps:
    :param current_frame:
    :return:
    '''
    # this function should change according to our selection of
    frame_selection = []
    last_selection = current_frame
    for i in range(current_frame, current_frame + time_steps):
        if (i >= frames_count):
            frame_selection.append(last_selection)
        else:
            frame_selection.append(i)
            last_selection = i

    return frame_selection


def get_nn_input(l_layer, resnet_out):
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
    total_frames = original_l_layers.shape[0]
    predicted_frames = []
    for i in range(total_frames):
        l_layer = original_l_layers[i]
        a_layer = np.multiply(predicted_AB_layers[i, 0, :, :, 0], 128)
        b_layer = np.multiply(predicted_AB_layers[i, 0, :, :, 1], 128)
        frame = np.empty((240, 320, 3))
        frame[:, :, 0] = l_layer
        frame[:, :, 1] = a_layer
        frame[:, :, 2] = b_layer
        frame = color.lab2rgb(frame)
        predicted_frames.append(frame)
    return predicted_frames


def save_output_video(frames, output_file):
    count = 0
    for frame in frames:
        cv2.imshow('image', frame)
        cv2.imwrite('/home/chamath/'+"img"+str(count)+".jpg", frame)
        count+=1
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # write to output_file


def manage_process(input_file, output_file):
    # read input video
    frames = get_video(input_file)

    # get L layer
    (original_l_layers, processed_l_layer) = get_l_layer(frames)
    print("Completed calculating L layer")

    # get resnet embeddings
    predictions = get_resnet_records(frames)
    print("Completed calculating resnet records ")

    # run flowchroma model
    ckpts = glob.glob("checkpoints/*.hdf5")
    if len(ckpts) == 0:
        print("No checkpoint found")
        exit(1)
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print("loading from checkpoint:", latest_ckpt)
    model = load_model(latest_ckpt, custom_objects={'FusionLayer': FusionLayer})
    X = get_nn_input(processed_l_layer, predictions)
    predictions = model.predict(X)
    print("Flowchroma model predictions calculated")

    # write output video file
    frame_predictions = post_process_predictions(original_l_layers, predictions)
    save_output_video(frame_predictions, output_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Sample video files')
    parser.add_argument('-i', '--input-file',
                        type=str,
                        metavar='INPUT',
                        dest='source',
                        help='file to convert')
    parser.add_argument('-o', '--output-file',
                        type=str,
                        metavar='OUTPUT',
                        dest='output',
                        help='output file')

    args = parser.parse_args()
    manage_process(args.source, args.output)
