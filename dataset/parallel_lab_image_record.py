from os import listdir
from os.path import join
from multiprocessing import Pool

import cv2
import numpy as np

from dataset.utils.resize import resize_pad_frame
from dataset.utils.shared import dir_sampled, dir_lab_records, default_nn_input_height, default_nn_input_width
from skimage import color

import re

input_dir = dir_sampled
output_dir = dir_lab_records
size = (default_nn_input_height, default_nn_input_width)
equal_padding = True


def write_to_csv(file, output_file):
    """
    Read content from AVI file and resize frames and convert frames to LAB color space
    :param file: video file
    :param output_file: LAB output file
    """
    video = cv2.VideoCapture(file)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = resize_pad_frame(frame, size, equal_padding=equal_padding)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        frame = color.rgb2lab(frame)
        frames.append(frame)
    frames = np.asarray(frames)

    # LAB layers should be brought to [-1, +1] region
    frames[:, :, :, 0] = np.divide(frames[:, :, :, 0], 50) - 1
    frames[:, :, :, 1] = np.divide(frames[:, :, :, 1], 128)
    frames[:, :, :, 2] = np.divide(frames[:, :, :, 2], 128)

    np.save(output_file, frames)


def prepare_file (file_name):
    m = re.search('video_([0-9]*).avi', file_name)
    if m:
        index = int(m.group(1))
    else:
        return
    input_file = join(input_dir, file_name)
    output_file = join(output_dir,"lab_record_" + format(index, '05d'))
    write_to_csv(input_file, output_file)
    print('Converted file: '+file_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Create lab records for video files')
    parser.add_argument('-s', '--source-folder',
                        type=str,
                        metavar='FOLDER',
                        default=dir_sampled,
                        dest='source',
                        help='use FOLDER as source of the videos')
    parser.add_argument('-o', '--output-folder',
                        type=str,
                        metavar='FILE',
                        default=dir_lab_records,
                        dest='output',
                        help='use FILE as destination')
    parser.add_argument('-t', '--height',
                        default=default_nn_input_height,
                        type=int,
                        metavar='HEIGHT',
                        dest='height',
                        help='use HEIGHT as height of frames')
    parser.add_argument('-w', '--width',
                        default=default_nn_input_width,
                        type=int,
                        metavar='WIDTH',
                        dest='width',
                        help='use WIDTH as width of a frame')
    parser.add_argument('-p', '--equal-padding',
                        default=True,
                        type=bool,
                        metavar='PAD',
                        dest='equal_padding',
                        help='use PAD to determine distribution of padding')

    args = parser.parse_args()
    input_dir = args.source
    output_dir = args.output
    size = (args.height, args.width)
    equal_padding = args.equal_padding

    file_list = listdir(input_dir)
    pool = Pool(8)
    pool.map(prepare_file, file_list)

